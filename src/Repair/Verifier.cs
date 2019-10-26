using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using GPURepair.Repair.Errors;
using GPURepair.Repair.Exceptions;
using GPURepair.Repair.Metadata;
using GPUVerify;
using Microsoft.Boogie;
using VC;

namespace GPURepair.Repair
{
    public class Verifier
    {
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

            KernelAnalyser.EliminateDeadVariables(this.program);
            KernelAnalyser.Inline(this.program);
            KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(this.program);
        }

        /// <summary>
        /// Gets the errors and associated metadata after verifying the program.
        /// </summary>
        /// <returns>The metadata of the errors.</returns>
        public IEnumerable<RepairableError> GetErrors()
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
                            errors.Add(GenerateError(example, implementation));
                }
            }

            gen.Close();
            if (!errors.Any(x => x is RepairableError && (x as RepairableError).Variables.Any()))
            {
                if (errors.Any(x => !(x is RepairableError)))
                    throw new NonBarrierError("The program cannot be repaired since it has errors besides race and divergence errors!");
                if (errors.Any(x => x.CounterExample is AssertCounterexample))
                    throw new AssertionError("Assertions do not hold!");
                if (errors.Any(x => x is RepairableError))
                    throw new RepairError("Encountered a counterexample without any barrier assignments!");
            }

            return errors.Where(x => x is RepairableError && (x as RepairableError).Variables.Any())
                .Select(x => x as RepairableError).ToList();
        }

        /// <summary>
        /// Generates the error from the counter example.
        /// </summary>
        /// <param name="example">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        /// <returns>The error.</returns>
        private Error GenerateError(Counterexample example, Implementation implementation)
        {
            if (example is CallCounterexample)
            {
                CallCounterexample callCounterexample = example as CallCounterexample;
                if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "barrier_divergence"))
                {
                    DivergenceError divergence = new DivergenceError(example, implementation);
                    IdentifyVariables(divergence);

                    return divergence;
                }
                else if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "race"))
                {
                    RaceError race = new RaceError(example, implementation);
                    IdentifyVariables(race);

                    return race;
                }
            }

            return new Error(example, implementation);
        }

        /// <summary>
        /// Identifies the variables from the counter example.
        /// </summary>
        /// <param name="error">The error.</param>
        private void IdentifyVariables(RepairableError error)
        {
            Regex regex = new Regex(@"^b\d+$");
            Dictionary<string, bool> assignments = new Dictionary<string, bool>();

            IEnumerable<Model.Boolean> elements = error.CounterExample.Model.Elements
                .Where(x => x is Model.Boolean).Select(x => x as Model.Boolean);
            foreach (Model.Boolean element in elements)
                element.References.Select(x => x.Func).Where(x => regex.IsMatch(x.Name))
                    .Select(x => x.Name).ToList().ForEach(x => assignments.Add(x, element.Value));

            foreach (string key in this.assignments.Keys)
                assignments.Add(key, this.assignments[key]);

            bool value = error is RaceError ? false : true;
            IEnumerable<string> names = assignments.Where(x => x.Value == value).Select(x => x.Key);

            IEnumerable<Barrier> barriers =
                ProgramMetadata.Barriers.Where(x => names.Contains(x.Key)).Select(x => x.Value);

            error.Variables = ProgramMetadata.Barriers.Where(x => names.Contains(x.Key)).Select(x => x.Value)
                .Where(x => x.Implementation.Name == error.Implementation.Name).Select(x => x.Variable);
        }
    }
}
