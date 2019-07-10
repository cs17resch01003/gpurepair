using GPUVerify;

namespace GPURepair.Repair
{
    public class GRCommandLineOptions : GVCommandLineOptions
    {
        public string RepairLog { get; set; } = null;

        public bool EnableEfficientSolving { get; set; } = true;

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
            else if (name == "efficientSolving")
            {
                if (ps.ConfirmArgumentCount(1))
                    EnableEfficientSolving = bool.Parse(ps.args[ps.i]);
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}