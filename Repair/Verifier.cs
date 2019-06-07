using GPURepair.Repair.Exceptions;
using GPUVerify;
using Microsoft.Boogie;
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
        /// Gets the errors and associated metadata after verifying the program.
        /// </summary>
        /// <returns>The metadata of the errors.</returns>
        public IEnumerable<Error> GetErrors()
        {
            List<Error> errors = new List<Error>();

            VCGen gen = new VCGen(program, CommandLineOptions.Clo.SimplifyLogFilePath, CommandLineOptions.Clo.SimplifyLogFileAppend, new List<Checker>());
            foreach (Declaration declaration in program.TopLevelDeclarations)
            {
                if (declaration is Implementation)
                {
                    Implementation implementation = declaration as Implementation;
                    List<Counterexample> examples;

                    ConditionGeneration.Outcome outcome = gen.VerifyImplementation(implementation, out examples);
                    if (outcome == ConditionGeneration.Outcome.Errors)
                        foreach (Counterexample example in examples)
                            errors.Add(new Error()
                            {
                                CounterExample = example,
                                Implementation = implementation
                            });
                }
            }

            List<Error> valid_errors = new List<Error>();
            foreach (Error error in errors)
            {
                if (error.CounterExample is CallCounterexample)
                {
                    CallCounterexample callCounterexample = error.CounterExample as CallCounterexample;
                    if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "barrier_divergence"))
                        error.ErrorType = ErrorType.Divergence;
                    else if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "race"))
                        error.ErrorType = ErrorType.Race;

                    if (error.ErrorType.HasValue)
                    {
                        IEnumerable<string> variables = GetVariables(callCounterexample, error.ErrorType.Value);
                        error.Variables = barriers.Where(x => variables.Contains(x.Key)).Select(x => x.Value);

                        if (error.Variables.Any())
                            valid_errors.Add(error);
                    }
                }
            }

            if (!valid_errors.Any() && errors.Any())
            {
                if (errors.Any(x => x.CounterExample is AssertCounterexample))
                    throw new AssertionError("Assertions do not hold!");
                if (errors.Any(x => x.CounterExample is CallCounterexample))
                    throw new RepairError("Encountered a race/divergence counterexample without any barrier assignments!");

                throw new NonBarrierError("The program cannot be repaired since it has errors besides race and divergence errors!");
            }

            gen.Close();
            return valid_errors;
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