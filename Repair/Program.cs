using GPUVerify;
using Microsoft.Boogie;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using VC;

namespace GPURepair.Repair
{
    public class Program
    {
        private static Dictionary<string, Variable> barriers = new Dictionary<string, Variable>();

        /// <summary>
        /// The entry point of the program.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            // standard command line options for Boogie
            CommandLineOptions.Install(new CommandLineOptions());
            if (!CommandLineOptions.Clo.Parse(args))
                return;

            CommandLineOptions.Clo.PrintUnstructured = 2;
            CommandLineOptions.Clo.DoModSetAnalysis = true;
            CommandLineOptions.Clo.PruneInfeasibleEdges = true;

            // hardcoded file name for testing purposes, has to be changed to use the command line argumnent
            string filePath = @"D:\Projects\GPURepair\kernel.cbpl";
            Microsoft.Boogie.Program program = ReadFile(filePath);

            PopulateBarriers(program, barriers);

            List<Error> errors = new List<Error>();
            while (true)
            {
                KernelAnalyser.EliminateDeadVariables(program);
                KernelAnalyser.Inline(program);
                KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(program);

                Error error = GetError(program);
                if (error == null)
                    break;

                errors.Add(error);

                program = ReadFile(filePath);
                ConstraintProgram(errors, program);

                string tempFile = filePath + "." + DateTime.Now.ToString("hhmmss") + ".temp";
                using (TokenTextWriter writer = new TokenTextWriter(tempFile, true))
                    program.Emit(writer);

                program = ReadFile(tempFile);
            }
        }

        private static Microsoft.Boogie.Program ReadFile(string filePath)
        {
            Microsoft.Boogie.Program program;
            Parser.Parse(filePath, new List<string>() { "FILE_0" }, out program);

            ResolutionContext context = new ResolutionContext(null);
            program.Resolve(context);

            program.Typecheck();
            new ModSetCollector().DoModSetAnalysis(program);

            return program;
        }

        /// <summary>
        /// Populates the references to global variables.
        /// </summary>
        /// <param name="program">The given program.</param>
        /// <param name="barriers">The barriers.</param>
        private static void PopulateBarriers(Microsoft.Boogie.Program program, Dictionary<string, Variable> barriers)
        {
            Regex regex = new Regex(@"^b\d+$");

            foreach (Declaration declaration in program.TopLevelDeclarations)
                if (declaration is GlobalVariable)
                {
                    GlobalVariable globalVariable = declaration as GlobalVariable;
                    if (regex.IsMatch(globalVariable.Name))
                        barriers.Add(globalVariable.Name, globalVariable);
                }
        }

        /// <summary>
        /// Gets the error and associated metadata after verifying the program.
        /// </summary>
        /// <param name="program">The given program.</param>
        /// <returns>The error metadata.</returns>
        private static Error GetError(Microsoft.Boogie.Program program)
        {
            VCGen gen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend, new List<Checker>());
            foreach (Declaration declaration in program.TopLevelDeclarations)
            {
                if (declaration is Implementation)
                {
                    Implementation implementation = declaration as Implementation;
                    List<Counterexample> errors;

                    ConditionGeneration.Outcome outcome = gen.VerifyImplementation(implementation, out errors);
                    if (outcome == ConditionGeneration.Outcome.Errors)
                    {
                        foreach (Counterexample counterexample in errors)
                        {
                            if (counterexample is CallCounterexample)
                            {
                                CallCounterexample callCounterexample = counterexample as CallCounterexample;
                                ErrorType errorType;

                                if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "barrier_divergence"))
                                    errorType = ErrorType.Divergence;
                                else if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "race"))
                                    errorType = ErrorType.Race;
                                else
                                    throw new Exception("The program cannot be repaired since it has errors besides race and divergence errors!");

                                IEnumerable<string> variables = GetVariables(callCounterexample, errorType);
                                return new Error
                                {
                                    ErrorType = errorType,
                                    Implementation = implementation,
                                    Variables = variables
                                };
                            }
                            else
                            {
                                throw new Exception("The program cannot be repaired since it has errors besides race and divergence errors!");
                            }
                        }
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// Gets the variables from the counter example.
        /// </summary>
        /// <param name="callCounterexample">The counter example.</param>
        /// <param name="errorType">The error type.</param>
        /// <returns></returns>
        private static IEnumerable<string> GetVariables(CallCounterexample callCounterexample, ErrorType errorType)
        {
            Model.Boolean element = callCounterexample.Model.Elements.Where(x => x is Model.Boolean)
                .Where(x => (x as Model.Boolean).Value == (errorType == ErrorType.Race ? false : true))
                .Select(x => x as Model.Boolean).FirstOrDefault();

            Regex regex = new Regex(@"^b\d+$");
            IEnumerable<string> variables = element.References.Select(x => x.Func)
                .Where(x => regex.IsMatch(x.Name)).Select(x => x.Name);

            return variables;
        }

        /// <summary>
        /// Applies the required constraints to the program.
        /// </summary>
        private static void ConstraintProgram(IEnumerable<Error> errors, Microsoft.Boogie.Program program)
        {
            // Use Z3 and figure out the variable assignments
            Dictionary<string, bool> assignments = new Dictionary<string, bool>();
            if (errors.Any())
                assignments.Add("b1", true);

            IEnumerable<Implementation> implementations = errors.Select(x => x.Implementation).Distinct();
            foreach (Implementation implementation in implementations)
            {
                Procedure procedure = program.Procedures.Where(x => x.Name == implementation.Name).First();

                IEnumerable<string> variables = errors.Where(x => x.Implementation == implementation)
                    .SelectMany(x => x.Variables).Distinct();
                IEnumerable<KeyValuePair<string, bool>> pairs = assignments.ToList().Where(x => variables.Contains(x.Key));

                ApplyAssignments(procedure, pairs);
            }
        }

        /// <summary>
        /// Apply the assignments in the form or requires statements.
        /// </summary>
        /// <param name="procedure">The procdeure.</param>
        /// <param name="pairs">The key-value pairs representing the variable name and it's value.</param>
        private static void ApplyAssignments(Procedure procedure, IEnumerable<KeyValuePair<string, bool>> pairs)
        {
            foreach (KeyValuePair<string, bool> pair in pairs)
            {
                procedure.Requires.Add(new Requires(false, new NAryExpr(Token.NoToken,
                    new BinaryOperator(Token.NoToken, BinaryOperator.Opcode.Eq), new List<Expr>()
                    {
                        new IdentifierExpr(Token.NoToken, barriers[pair.Key]),
                        new LiteralExpr(Token.NoToken, pair.Value)
                    })));
            }
        }
    }
}