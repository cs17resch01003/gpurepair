namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Repair.Metadata;
    using GPUVerify;

    public static class ExtensionMethods
    {
        public static IEnumerable<LocationChain> GetLocationChain(this SourceLocationInfo source)
        {
            List<LocationChain> chains = new List<LocationChain>();
            foreach (LocationChain chain in ProgramMetadata.Locations.Values)
            {
                if (chain.Equals(source))
                {
                    chains.Add(chain);
                    return chains;
                }
            }

            foreach (LocationChain chain in ProgramMetadata.Locations.Values)
                if (chain.Last().Equals(source.GetRecords().Last()))
                    chains.Add(chain);

            return chains;
        }
    }
}
