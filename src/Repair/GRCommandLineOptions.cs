namespace GPURepair.Repair
{
    using System;
    using GPUVerify;

    public class GRCommandLineOptions : GVCommandLineOptions
    {
        /// <summary>
        /// Enables additional logging.
        /// </summary>
        public bool AdditionalLogging { get; set; } = false;

        /// <summary>
        /// Enables logging of clauses.
        /// </summary>
        public bool LogClauses { get; set; } = false;

        /// <summary>
        /// The solver type.
        /// </summary>
        public Solver.SolverType SolverType = Solver.SolverType.MHS;

        /// <summary>
        /// Initializes an instance of <see cref="GRCommandLineOptions"/>.
        /// </summary>
        public GRCommandLineOptions()
            : base()
        {
        }

        /// <summary>
        /// Parses the command line options.
        /// </summary>
        /// <param name="name">The name of the option.</param>
        /// <param name="ps">The command line parsing state.</param>
        /// <returns>True if the parsing was successful.</returns>
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
