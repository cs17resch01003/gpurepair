﻿namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Repair.Errors;
    using GPUVerify;
    using Microsoft.Boogie;

    public class ErrorReporter : GPUVerifyErrorReporter
    {
        /// <summary>
        /// Instantiates an instance of <see cref="ErrorReporter"/>.
        /// </summary>
        /// <param name="program">The program.</param>
        /// <param name="implName">The name of the implementation.</param>
        public ErrorReporter(Microsoft.Boogie.Program program, string implName) : base(program, implName)
        {
        }

        /// <summary>
        /// Retrieves the race information.
        /// </summary>
        /// <param name="example">The counter example.</param>
        /// <param name="error">The race error.</param>
        /// <returns>The race errors.</returns>
        public IEnumerable<RaceError> GetRaceInformation(CallCounterexample example, RaceError error)
        {
            PopulateModelWithStatesIfNecessary(example);

            string raceyArrayName = GetArrayName(example.FailingRequires);
            IEnumerable<SourceLocationInfo> possibleSourcesForFirstAccess =
                GetPossibleSourceLocationsForFirstAccessInRace(
                    example, raceyArrayName, GetAccessType(example), GetStateName(example));
            SourceLocationInfo sourceInfoForSecondAccess = new SourceLocationInfo(
                example.FailingCall.Attributes, GetSourceFileName(), example.FailingCall.tok);

            List<RaceError> errors = new List<RaceError>();
            foreach (SourceLocationInfo possibleSourceForFirstAccess in possibleSourcesForFirstAccess)
            {
                RaceError race = new RaceError(error.CounterExample, error.Implementation)
                {
                    Start = possibleSourceForFirstAccess,
                    End = sourceInfoForSecondAccess
                };

                errors.Add(race);
            }

            if (!errors.Any())
                return new List<RaceError> { error };
            return errors;
        }

        /// <summary>
        /// Populates the divergence information.
        /// </summary>
        /// <param name="example">The counter example.</param>
        /// <param name="error">The divergence error.</param>
        public void PopulateDivergenceInformation(CallCounterexample example, DivergenceError error)
        {
            CallCmd call = example.FailingCall;
            SourceLocationInfo source = new SourceLocationInfo(call.Attributes, GetSourceFileName(), call.tok);
            error.Location = source;
        }

        private static string GetSourceFileName()
        {
            return CommandLineOptions.Clo.Files[CommandLineOptions.Clo.Files.Count() - 1];
        }
    }
}
