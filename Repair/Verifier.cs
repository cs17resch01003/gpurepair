using GPURepair.Repair.Exceptions;
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
        /// The current assignments to the variables.
        /// </summary>
        private Dictionary<string, bool> assignments;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        public Verifier(Microsoft.Boogie.Program program, Dictionary<string, bool> assignments)
        {
            this.program = program;
            this.assignments = assignments;

            if (barriers == null)
                PopulateBarriers(this.program);

            KernelAnalyser.EliminateDeadVariables(this.program);
            KernelAnalyser.Inline(this.program);
            KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(this.program);
        }

        /// <summary>
        /// Gets the error and associated metadata after verifying the program.
        /// </summary>
        /// <returns>The metadata of the errors.</returns>
        public IEnumerable<Error> GetError()
        {
            List<Error> errors = new List<Error>();

            VCGen gen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend, new List<Checker>());
            foreach (Declaration declaration in program.TopLevelDeclarations)
            {
                if (declaration is Implementation)
                {
                    Implementation implementation = declaration as Implementation;
                    List<Counterexample> counterexamples;

                    ConditionGeneration.Outcome outcome = gen.VerifyImplementation(implementation, out counterexamples);
                    if (outcome == ConditionGeneration.Outcome.Errors)
                    {
                        foreach (Counterexample counterexample in counterexamples)
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
                                    throw new NonBarrierError("The program cannot be repaired since it has errors besides race and divergence errors!");

                                IEnumerable<string> variables = GetVariables(callCounterexample, errorType);
                                errors.Add(new Error
                                {
                                    ErrorType = errorType,
                                    Implementation = implementation,
                                    Variables = barriers.Where(x => variables.Contains(x.Key)).Select(x => x.Value)
                                });

                                if (errors.Where(x => !x.Variables.Any()).Any())
                                    throw new RepairError("Encountered a race/divergence counterexample without any barrier assignments!");
                            }
                            else if (counterexamples.Count == 1)
                            {
                                throw new NonBarrierError("The program cannot be repaired since it has errors besides race and divergence errors!");
                            }
                        }
                    }
                }
            }

            return errors;
        }

        /// <summary>
        /// Gets the variables from the counter example.
        /// </summary>
        /// <param name="callCounterexample">The counter example.</param>
        /// <param name="errorType">The error type.</param>
        /// <returns></returns>
        private IEnumerable<string> GetVariables(CallCounterexample callCounterexample, ErrorType errorType)
        {
            Regex regex = new Regex(@"^b\d+$");
            Model.Boolean element = callCounterexample.Model.Elements.Where(x => x is Model.Boolean)
                .Where(x => (x as Model.Boolean).Value == (errorType == ErrorType.Race ? false : true))
                .Select(x => x as Model.Boolean).FirstOrDefault();

            IEnumerable<string> variables = element.References.Select(x => x.Func)
                .Where(x => regex.IsMatch(x.Name)).Select(x => x.Name);
            IEnumerable<string> current_assignments = assignments.Where(x => x.Value == ((errorType == ErrorType.Race ? false : true)))
                .Select(x => x.Key);

            return variables.Union(current_assignments);
        }
    }
}