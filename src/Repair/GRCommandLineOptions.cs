using System;
using GPUVerify;

namespace GPURepair.Repair
{
    public class GRCommandLineOptions : GVCommandLineOptions
    {
        public bool AdditionalLogging { get; set; } = false;

        public bool LogClauses { get; set; } = false;

        public Solver.SolverType SolverType = Solver.SolverType.MHS;

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
            else if (name == "logClauses")
            {
                if (ps.ConfirmArgumentCount(1))
                    LogClauses = bool.Parse(ps.args[ps.i]);
                return true;
            }
            else if (name == "solverType")
            {
                if (ps.ConfirmArgumentCount(1))
                    SolverType = (Solver.SolverType)Enum.Parse(typeof(Solver.SolverType), ps.args[ps.i]);
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
