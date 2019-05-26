using GPUVerify;

namespace GPURepair.Repair
{
    public class GRCommandLineOptions : GVCommandLineOptions
    {
        public string TimeLog { get; set; }

        public GRCommandLineOptions()
            : base()
        {
        }

        protected override bool ParseOption(string name, CommandLineParseState ps)
        {
            if (name == "timeLog")
            {
                if (ps.ConfirmArgumentCount(1))
                    TimeLog = ps.args[ps.i];
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}