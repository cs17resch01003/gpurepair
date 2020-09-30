﻿namespace GPURepair.Repair
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
            if (error is RaceError)
            {
                PopulateRaceBarriers(barriers, error as RaceError);
            }
            else
            {
                DivergenceError divergence = error as DivergenceError;
                List<Barrier> final_barriers = new List<Barrier>();

                foreach (Barrier barrier in barriers)
                {
                    List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];
                    foreach (Location location in locations)
                    {
                        if (divergence.Location != null && location.Equals(divergence.Location))
                            AddBarrier(final_barriers, barrier);
                    }
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

            List<Barrier> location_barriers = FilterBarriers(barriers, race.Start, race.End);
            if (!location_barriers.Any())
            {
                location_barriers = FilterBarriers(race.End, race.Start);
                inverse = true;
            }

            // get all the other barriers which are in the loop
            List<Barrier> loop_barriers = new List<Barrier>();
            foreach (BackEdge edge in ProgramMetadata.LoopBarriers.Keys)
            {
                List<Barrier> temp = ProgramMetadata.LoopBarriers[edge];
                if (location_barriers.Any(x => temp.Contains(x)))
                    foreach (Barrier barrier in temp)
                        loop_barriers.Add(barrier);
            }

            if (inverse)
            {
                List<Barrier> inverse_barriers = new List<Barrier>();
                foreach (Barrier barrier in loop_barriers)
                    if (!location_barriers.Contains(barrier))
                        inverse_barriers.Add(barrier);

                location_barriers = inverse_barriers;
            }

            List<Barrier> result = new List<Barrier>();
            foreach (Barrier barrier in location_barriers)
                if (barriers.Contains(barrier))
                    AddBarrier(result, barrier);

            List<Barrier> overapproximated_result = new List<Barrier>();
            foreach (Barrier barrier in loop_barriers)
                if (barriers.Contains(barrier))
                    AddBarrier(overapproximated_result, barrier);

            race.Barriers = result;
            race.OverapproximatedBarriers = overapproximated_result;
        }

        /// <summary>
        /// Filter the barriers based on the source information.
        /// </summary>
        /// <param name="barriers">The barriers involved in the trace.</param>
        /// <param name="start">The start location.</param>
        /// <param name="end">The end location.</param>
        /// <returns>The barriers to be considered.</returns>
        private List<Barrier> FilterBarriers(IEnumerable<Barrier> barriers, Location start, Location end)
        {
            List<Barrier> location_barriers = new List<Barrier>();
            foreach (Barrier barrier in barriers)
            {
                List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];
                foreach (Location location in locations)
                {
                    if (start != null && end != null)
                        if (location.IsBetween(start, end))
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
        private List<Barrier> FilterBarriers(Location start, Location end)
        {
            List<Barrier> location_barriers = new List<Barrier>();
            foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
            {
                List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];
                foreach (Location location in locations)
                {
                    if (start != null && end != null)
                        if (location.IsBetween(start, end))
                            AddBarrier(location_barriers, barrier);
                }
            }

            return location_barriers;
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
