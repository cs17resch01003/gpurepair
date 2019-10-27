using System.Collections.Generic;
using System.Linq;
using GPURepair.Repair.Errors;
using GPURepair.Repair.Metadata;
using GPUVerify;
using Microsoft.Boogie;

namespace GPURepair.Repair
{
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
        /// Populates the race information.
        /// </summary>
        /// <param name="example">The counter example.</param>
        /// <param name="error">The race error.</param>
        public void PopulateRaceInformation(CallCounterexample example, RaceError error)
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

            error.RaceType = raceName;
            error.Access1 = access1;
            error.Access2 = access2;

            SourceLocationInfo.Record second = sourceInfoForSecondAccess.GetRecords().Last();
            Location second_location = ProgramMetadata.GetLocation(
                second.GetDirectory(), second.GetFile(), second.GetLine(), second.GetColumn());

            List<Location> possible_first_locations = new List<Location>();
            foreach (SourceLocationInfo source in possibleSourcesForFirstAccess)
            {
                SourceLocationInfo.Record record = source.GetRecords().Last();
                if (record.GetDirectory() == second.GetDirectory() && record.GetFile() == second.GetFile())
                {
                    Location location = ProgramMetadata.GetLocation(
                        record.GetDirectory(), record.GetFile(), record.GetLine(), record.GetColumn());
                    if (location != null)
                        possible_first_locations.Add(location);
                }
            }

            Location first_location = possible_first_locations.FirstOrDefault();
            foreach (Location possible_first_location in possible_first_locations)
                if (possible_first_location.ComesBefore(first_location))
                    first_location = possible_first_location;

            if (first_location != null)
            {
                error.Start = first_location;
                error.End = second_location;
            }
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

            SourceLocationInfo.Record record = source.GetRecords().Last();
            Location location = ProgramMetadata.GetLocation(
                record.GetDirectory(), record.GetFile(), record.GetLine(), record.GetColumn());

            error.Location = location;
        }
    }
}
