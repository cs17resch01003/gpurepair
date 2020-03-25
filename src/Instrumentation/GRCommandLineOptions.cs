using Microsoft.Boogie;

namespace GPURepair.Instrumentation
{
    public class GRCommandLineOptions : CommandLineOptions
    {
        public string InstrumentationLog { get; set; } = null;

        public GRCommandLineOptions()
            : base()
        {
        }

        protected override bool ParseOption(string name, CommandLineParseState ps)
        {
            if (name == "instrumentationLog")
            {
                if (ps.ConfirmArgumentCount(1))
                    InstrumentationLog = ps.args[ps.i];
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
