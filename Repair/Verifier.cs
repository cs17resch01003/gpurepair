using GPUVerify;
using Microsoft.Boogie;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using VC;

namespace GPURepair.Repair
{
    public class Verifier
    {
        /// <summary>
        /// The barriers in the program.
        /// </summary>
        private static Dictionary<string, Variable> barriers;

        /// <summary>
        /// Populates the barriers.
        /// </summary>
        /// <param name="program">The given program.</param>
        private static void PopulateBarriers(Microsoft.Boogie.Program program)
        {
            barriers = new Dictionary<string, Variable>();
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
        /// The program to be instrumented.
        /// </summary>
        private Microsoft.Boogie.Program program;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public Verifier(Microsoft.Boogie.Program program)
        {
            this.program = program;
            if (barriers == null)
                PopulateBarriers(this.program);

            KernelAnalyser.EliminateDeadVariables(this.program);
            KernelAnalyser.Inline(this.program);
            KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(this.program);
        }

        /// <summary>
        /// Gets the error and associated metadata after verifying the program.
        /// </summary>
        /// <returns>The error metadata.</returns>
        public Error GetError()
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
                                    Variables = barriers.Where(x => variables.Contains(x.Key)).Select(x => x.Value)
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
        private IEnumerable<string> GetVariables(CallCounterexample callCounterexample, ErrorType errorType)
        {
            Model.Boolean element = callCounterexample.Model.Elements.Where(x => x is Model.Boolean)
                .Where(x => (x as Model.Boolean).Value == (errorType == ErrorType.Race ? false : true))
                .Select(x => x as Model.Boolean).FirstOrDefault();

            Regex regex = new Regex(@"^b\d+$");
            IEnumerable<string> variables = element.References.Select(x => x.Func)
                .Where(x => regex.IsMatch(x.Name)).Select(x => x.Name);

            return variables;
        }
    }
}