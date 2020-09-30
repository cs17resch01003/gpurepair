namespace GPURepair.Repair
{
    using System.Collections.Generic;
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
            string raceName, access1, access2;

            DetermineNatureOfRace(example, out raceName, out access1, out access2);
            PopulateModelWithStatesIfNecessary(example);

            string raceyArrayName = GetArrayName(example.FailingRequires);
            IEnumerable<SourceLocationInfo> possibleSourcesForFirstAccess =
                GetPossibleSourceLocationsForFirstAccessInRace(
                    example, raceyArrayName, AccessType.Create(access1), GetStateName(example));
            SourceLocationInfo sourceInfoForSecondAccess = new SourceLocationInfo(
                GetAttributes(example.FailingCall), GetSourceFileName(), example.FailingCall.tok);

            List<RaceError> errors = new List<RaceError>();
            foreach (SourceLocationInfo possibleSourceForFirstAccess in possibleSourcesForFirstAccess)
            {
                RaceError race = new RaceError(error.CounterExample, error.Implementation)
                {
                    RaceType = raceName,
                    Access1 = access1,
                    Access2 = access2,
                    Start = possibleSourceForFirstAccess,
                    End = sourceInfoForSecondAccess
                };

                errors.Add(race);
            }

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
            SourceLocationInfo source = new SourceLocationInfo(GetAttributes(call), GetSourceFileName(), call.tok);
            error.Location = source;
        }
    }
}
