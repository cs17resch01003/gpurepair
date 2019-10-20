using GPUVerify;

namespace GPURepair.Repair
{
    public class GRCommandLineOptions : GVCommandLineOptions
    {
        public string RepairLog { get; set; } = null;

        public bool LogClauses { get; set; } = false;

        public GRCommandLineOptions()
            : base()
        {
        }

        protected override bool ParseOption(string name, CommandLineParseState ps)
        {
            if (name == "repairLog")
            {
                if (ps.ConfirmArgumentCount(1))
                    RepairLog = ps.args[ps.i];
                return true;
            }
            else if (name == "logClauses")
            {
                if (ps.ConfirmArgumentCount(1))
                    LogClauses = bool.Parse(ps.args[ps.i]);
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
