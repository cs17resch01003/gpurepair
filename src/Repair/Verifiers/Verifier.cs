namespace GPURepair.Repair.Verifiers
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

    public abstract class Verifier
    {
        /// <summary>
        /// The program to be instrumented.
        /// </summary>
        protected Program program;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public Verifier(Program program)
        {
            this.program = program;

            KernelAnalyser.EliminateDeadVariables(this.program);
            KernelAnalyser.Inline(this.program);
            KernelAnalyser.CheckForQuantifiersAndSpecifyLogic(this.program);
        }

        /// <summary>
        /// Gets the errors and associated metadata after verifying the program.
        /// </summary>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The metadata of the errors.</returns>
        public IEnumerable<RepairableError> GetErrors(Dictionary<string, bool> assignments)
        {
            List<Error> errors = new List<Error>();
            foreach (Declaration declaration in program.TopLevelDeclarations)
            {
                if (declaration is Implementation)
                {
                    Implementation implementation = declaration as Implementation;
                    if (!implementation.SkipVerification)
                    {
                        IEnumerable<Counterexample> examples = VerifyImplementation(implementation, assignments);
                        foreach (Counterexample example in examples)
                            errors.AddRange(GenerateErrors(example, implementation, assignments));
                    }
                }
            }

            // there are no repairable errors that have a variable assigned to them
            if (!errors.Any(x => x is RepairableError && (x as RepairableError).Barriers.Any()))
            {
                if (errors.Any(x => x.CounterExample is AssertCounterexample))
                    throw new AssertionException("Assertions do not hold!");
                if (errors.Any(x => !(x is RepairableError)))
                    throw new NonBarrierException("The program cannot be repaired since it has errors besides race and divergence errors!");
                if (errors.Any(x => x is RepairableError))
                    throw new RepairException("Encountered a counterexample without any barrier assignments!");
            }

            return errors.Where(x => x is RepairableError && (x as RepairableError).Barriers.Any())
                .Select(x => x as RepairableError).ToList();
        }

        /// <summary>
        /// Gets the counter examples after verifying the program.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The counter examples.</returns>
        protected abstract IEnumerable<Counterexample> VerifyImplementation(
            Implementation implementation, Dictionary<string, bool> assignments);

        /// <summary>
        /// Generates the errors from the counter example.
        /// </summary>
        /// <param name="example">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The errors.</returns>
        private IEnumerable<Error> GenerateErrors(Counterexample example, Implementation implementation,
            Dictionary<string, bool> assignments)
        {
            List<Error> errors = new List<Error>();
            if (example is CallCounterexample)
            {
                CallCounterexample callCounterexample = example as CallCounterexample;
                if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "barrier_divergence"))
                {
                    DivergenceError divergence = new DivergenceError(example, implementation);
                    ErrorReporter reporter = new ErrorReporter(program, implementation.Name);

                    reporter.PopulateDivergenceInformation(callCounterexample, divergence);
                    IdentifyVariables(divergence, assignments);

                    errors.Add(divergence);
                    return errors;
                }
                else if (QKeyValue.FindBoolAttribute(callCounterexample.FailingRequires.Attributes, "race"))
                {
                    RaceError template = new RaceError(example, implementation);
                    ErrorReporter reporter = new ErrorReporter(program, implementation.Name);

                    IEnumerable<RaceError> races = reporter.GetRaceInformation(callCounterexample, template);
                    foreach (RaceError race in races)
                        IdentifyVariables(race, assignments);

                    errors.AddRange(races);
                    return errors;
                }
            }

            errors.Add(new Error(example, implementation));
            return errors;
        }

        /// <summary>
        /// Identifies the variables from the counter example.
        /// </summary>
        /// <param name="error">The error.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        private void IdentifyVariables(RepairableError error, Dictionary<string, bool> assignments)
        {
            // Filter the variables based on source locations
            if (error is RaceError)
            {
                Regex regex = new Regex(@"^b\d+$");
                Dictionary<string, bool> local = new Dictionary<string, bool>();

                IEnumerable<Model.Boolean> elements = error.CounterExample.Model.Elements
                    .Where(x => x is Model.Boolean).Select(x => x as Model.Boolean);
                foreach (Model.Boolean element in elements)
                    element.References.Select(x => x.Func).Where(x => regex.IsMatch(x.Name))
                        .Select(x => x.Name).ToList().ForEach(x => local.Add(x, element.Value));

                foreach (string key in assignments.Keys)
                {
                    if (local.ContainsKey(key))
                    {
                        if (local[key] != assignments[key])
                            throw new RepairException("The model value doesn't match the assignment value!");
                    }
                    else
                    {
                        local.Add(key, assignments[key]);
                    }
                }

                bool value = error is RaceError ? false : true;
                IEnumerable<string> names = local.Where(x => x.Value == value).Select(x => x.Key);

                IEnumerable<Barrier> barriers = ProgramMetadata.Barriers
                    .Where(x => names.Contains(x.Key)).Select(x => x.Value)
                    .Where(x => x.Implementation.Name == error.Implementation.Name);

                PopulateRaceBarriers(barriers, error as RaceError);
            }
            else
            {
                DivergenceError divergence = error as DivergenceError;
                List<Barrier> final_barriers = new List<Barrier>();

                foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                {
                    LocationChain chain = ProgramMetadata.Locations[barrier.SourceLocation];
                    if (chain.Equals(divergence.Location))
                        AddBarrier(final_barriers, barrier);
                }

                divergence.Barriers = final_barriers;
            }
        }

        /// <summary>
        /// Populates the barriers based on the source information.
        /// </summary>
        /// <param name="barriers">The barriers involved in the trace.</param>
        /// <param name="race">The race error.</param>
        private void PopulateRaceBarriers(IEnumerable<Barrier> barriers, RaceError race)
        {
            bool inverse = false;

            List<Barrier> location_barriers;
            if (race.Start == null && race.End == null)
            {
                location_barriers = barriers.ToList();
            }
            else
            {
                location_barriers = FilterBarriers(barriers, race.Start, race.End);
                if (!location_barriers.Any())
                {
                    location_barriers = FilterBarriers(race.End, race.Start);
                    inverse = true;
                }
            }

            // get all the other barriers which are in the loop
            List<Barrier> loop_barriers = GetLoopBarriers(race.Start, race.End);
            if (inverse)
            {
                List<Barrier> inverse_barriers = new List<Barrier>();
                foreach (Barrier barrier in loop_barriers)
                    if (!location_barriers.Contains(barrier))
                        inverse_barriers.Add(barrier);

                location_barriers = inverse_barriers;
            }

            // check if all the locations are from the loop
            // needed for cases for one location is inside and the other is outside the loop
            // refer CUDASamples\6_Advanced_concurrentKernels_sum as an example
            bool same_loop = !location_barriers.Any(x => !loop_barriers.Contains(x));

            List<Barrier> result = new List<Barrier>();
            foreach (Barrier barrier in location_barriers)
                if (barriers.Contains(barrier))
                    AddBarrier(result, barrier);

            List<Barrier> overapproximated_result = new List<Barrier>();
            foreach (Barrier barrier in loop_barriers.Union(location_barriers))
                if (barriers.Contains(barrier))
                    AddBarrier(overapproximated_result, barrier);

            race.Barriers = same_loop ? result : overapproximated_result;
            race.OverapproximatedBarriers = overapproximated_result;
        }

        /// <summary>
        /// Filter the barriers based on the source information.
        /// </summary>
        /// <param name="barriers">The barriers involved in the trace.</param>
        /// <param name="start">The start location.</param>
        /// <param name="end">The end location.</param>
        /// <returns>The barriers to be considered.</returns>
        private List<Barrier> FilterBarriers(
            IEnumerable<Barrier> barriers, SourceLocationInfo start, SourceLocationInfo end)
        {
            List<Barrier> location_barriers = new List<Barrier>();
            foreach (Barrier barrier in barriers)
            {
                LocationChain chain = ProgramMetadata.Locations[barrier.SourceLocation];
                if (start != null && end != null)
                {
                    if (chain.IsBetween(start, end))
                        AddBarrier(location_barriers, barrier);
                }
            }

            return location_barriers;
        }

        /// <summary>
        /// Filter the barriers based on the source information.
        /// </summary>
        /// <param name="start">The start location.</param>
        /// <param name="end">The end location.</param>
        /// <returns>The barriers to be considered.</returns>
        private List<Barrier> FilterBarriers(SourceLocationInfo start, SourceLocationInfo end)
        {
            List<Barrier> location_barriers = new List<Barrier>();
            foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
            {
                LocationChain chain = ProgramMetadata.Locations[barrier.SourceLocation];
                if (start != null && end != null)
                {
                    if (chain.IsBetween(start, end))
                        AddBarrier(location_barriers, barrier);
                }
            }

            return location_barriers;
        }

        /// <summary>
        /// Adds a barrier to the list if it doesn't exist already.
        /// </summary>
        /// <param name="list">The list.</param>
        /// <param name="barrier">The barrier.</param>
        private void AddBarrier(List<Barrier> list, Barrier barrier)
        {
            if (!list.Contains(barrier))
                list.Add(barrier);
        }

        /// <summary>
        /// Gets the loop barriers if any of the given locations are inside a loop.
        /// </summary>
        /// <param name="start">The starting location.</param>
        /// <param name="end">The ending location.</param>
        /// <returns>The loop barriers.</returns>
        private List<Barrier> GetLoopBarriers(SourceLocationInfo start, SourceLocationInfo end)
        {
            List<Barrier> result = new List<Barrier>();
            foreach (BackEdge edge in ProgramMetadata.LoopBarriers.Keys)
            {
                List<Barrier> loop_barriers = ProgramMetadata.LoopBarriers[edge];
                IEnumerable<LocationChain> locations =
                    loop_barriers.Select(x => ProgramMetadata.Locations[x.SourceLocation]);

                foreach (LocationChain chain in locations)
                {
                    if (chain.Equals(start) || chain.Equals(end))
                    {
                        result.AddRange(loop_barriers);
                        break;
                    }
                }
            }

            return result;
        }
    }
}
