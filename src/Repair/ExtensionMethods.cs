namespace GPURepair.Repair
{
    using GPURepair.Repair.Metadata;
    using GPUVerify;

    public static class ExtensionMethods
    {
        public static LocationChain GetLocationChain(this SourceLocationInfo source)
        {
            foreach (LocationChain chain in ProgramMetadata.Locations.Values)
                if (chain.Equals(source))
                    return chain;

            return null;
        }
    }
}
