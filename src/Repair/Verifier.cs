namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using System.Text.RegularExpressions;
    using GPURepair.Common;
    using GPURepair.Repair.Errors;
    using GPURepair.Repair.Exceptions;
    using GPURepair.Repair.Metadata;
    using GPUVerify;
    using Microsoft.Boogie;
    using VC;

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

            // there are no repairable errors that have a variable assigned to them
            if (!errors.Any(x => x is RepairableError && (x as RepairableError).Variables.Any()))
            {
                if (errors.Any(x => x.CounterExample is AssertCounterexample))
                    throw new AssertionException("Assertions do not hold!");
                if (errors.Any(x => !(x is RepairableError)))
                    throw new NonBarrierException("The program cannot be repaired since it has errors besides race and divergence errors!");
                if (errors.Any(x => x is RepairableError))
                    throw new RepairException("Encountered a counterexample without any barrier assignments!");
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
                    ErrorReporter reporter = new ErrorReporter(program, implementation.Name);

                    reporter.PopulateDivergenceInformation(callCounterexample, divergence);
                    IdentifyVariables(divergence);

                    return divergence;
                }
                else if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "race"))
                {
                    RaceError race = new RaceError(example, implementation);
                    ErrorReporter reporter = new ErrorReporter(program, implementation.Name);

                    reporter.PopulateRaceInformation(callCounterexample, race);
                    IdentifyVariables(race);

                    return race;
                }
            }

            // enabling certain barriers could cause assertion errors
            else if (example is AssertCounterexample)
            {
                AssertionError assertion = new AssertionError(example, implementation);
                IEnumerable<string> names = assignments.Where(x => x.Value == true).Select(x => x.Key);

                IEnumerable<Barrier> barriers = ProgramMetadata.Barriers
                    .Where(x => names.Contains(x.Key)).Select(x => x.Value);
                assertion.Variables = barriers.Select(x => x.Variable).ToList();

                return assertion;
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

            IEnumerable<Barrier> barriers = ProgramMetadata.Barriers
                .Where(x => names.Contains(x.Key)).Select(x => x.Value)
                .Where(x => x.Implementation.Name == error.Implementation.Name);

            // Filter the variables based on source locations
            List<Variable> variables = new List<Variable>();
            if (error is RaceError)
            {
                List<Barrier> filteredBarriers = FilterBarriers(barriers, error as RaceError);
                variables = filteredBarriers.Select(x => x.Variable).ToList();
            }
            else
            {
                DivergenceError divergence = error as DivergenceError;
                foreach (Barrier barrier in barriers)
                {
                    List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];
                    foreach (Location location in locations)
                    {
                        if (divergence.Location != null && location.Equals(divergence.Location))
                            variables.Add(barrier.Variable);
                    }
                }
            }

            if (!variables.Any() && barriers.Any())
                variables = barriers.Select(x => x.Variable).ToList();
            error.Variables = variables;
        }

        /// <summary>
        /// Filter the barriers based on the source information.
        /// </summary>
        /// <param name="barriers">The barriers involved in the trace.</param>
        /// <param name="race">The race error.</param>
        /// <returns>The barriers to be considered.</returns>
        private List<Barrier> FilterBarriers(IEnumerable<Barrier> barriers, RaceError race)
        {
            List<Barrier> filteredBarriers = new List<Barrier>();
            foreach (Barrier barrier in barriers)
            {
                List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];
                foreach (Location location in locations)
                {
                    if (race.Start != null && race.End != null)
                        if (location.IsBetween(race.Start, race.End))
                            filteredBarriers.Add(barrier);
                }
            }

            // check if any of the barriers are inside a loop
            List<Barrier> result = new List<Barrier>();
            foreach (BackEdge edge in ProgramMetadata.LoopBarriers.Keys)
            {
                List<Barrier> loopBarriers = ProgramMetadata.LoopBarriers[edge];
                if (filteredBarriers.Any(x => loopBarriers.Contains(x)))
                    foreach (Barrier barrier in loopBarriers)
                        if (barriers.Contains(barrier))
                            AddBarrier(result, barrier);
            }

            foreach (Barrier barrier in filteredBarriers)
                AddBarrier(result, barrier);

            return result;
        }

        /// <summary>
        /// Adds a barrier to the list if it doesn't exist already..
        /// </summary>
        /// <param name="list">The list.</param>
        /// <param name="barrier">The barrier.</param>
        private void AddBarrier(List<Barrier> list, Barrier barrier)
        {
            if (!list.Contains(barrier))
                list.Add(barrier);
        }
    }
}
