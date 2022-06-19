namespace GPURepair.Repair.Metadata
{
    using System.Collections.Generic;
    using System.Linq;
    using GPUVerify;

    public class LocationChain : List<Location>
    {
        /// <summary>
        /// The source location.
        /// </summary>
        public int SourceLocation { get; set; }

        public bool Equals(SourceLocationInfo source)
        {
            return SourceLocation == source.SourceLocation;
        }

        public bool IsBetween(SourceLocationInfo start, SourceLocationInfo end)
        {
            LocationChain start_chain = ProgramMetadata.Locations[start.SourceLocation.Value];
            LocationChain end_chain = ProgramMetadata.Locations[end.SourceLocation.Value];

            return IsBetween(start_chain, end_chain);
        }

        public bool IsBetween(LocationChain start, LocationChain end)
        {
            bool first = this.First().IsBetween(start.First(), end.First());
            bool last = this.Last().IsBetween(start.Last(), end.Last());

            return first || last;
        }

        /// <summary>
        /// Represents the location in a string format.
        /// </summary>
        /// <returns>A string representation of the location.</returns>
        public override string ToString()
        {
            return string.Join(" => ", this.Select(x => x.ToString()));
        }
    }
}
