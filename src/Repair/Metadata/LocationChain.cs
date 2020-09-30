namespace GPURepair.Repair.Metadata
{
    using System.Collections.Generic;
    using System.Linq;
    using GPUVerify;

    public class LocationChain : List<Location>
    {
        public bool Equals(SourceLocationInfo source)
        {
            if (Count == source.GetRecords().Count)
            {
                bool result = true;
                for (int i = 0; i < Count; i++)
                    result = result && this[i].Equals(source.GetRecords()[i]);
                return result;
            }

            return false;
        }

        public bool IsBetween(SourceLocationInfo start, SourceLocationInfo end)
        {
            IEnumerable<LocationChain> start_chains = start.GetLocationChain();
            IEnumerable<LocationChain> end_chains = end.GetLocationChain();

            foreach (LocationChain start_chain in start_chains)
                foreach (LocationChain end_chain in end_chains)
                    if (IsBetween(start_chain, end_chain))
                        return true;

            return false;
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
