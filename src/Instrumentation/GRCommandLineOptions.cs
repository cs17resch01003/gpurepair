using Microsoft.Boogie;

namespace GPURepair.Instrumentation
{
    public class GRCommandLineOptions : CommandLineOptions
    {
        public bool AdditionalLogging { get; set; } = false;

        public GRCommandLineOptions()
            : base()
        {
        }

        protected override bool ParseOption(string name, CommandLineParseState ps)
        {
            if (name == "additionalLogging")
            {
                if (ps.ConfirmArgumentCount(1))
                    AdditionalLogging = bool.Parse(ps.args[ps.i]);
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
