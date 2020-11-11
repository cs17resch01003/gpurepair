namespace GPURepair.Instrumentation
{
    using GPURepair.Common;
    using Microsoft.Boogie;

    public class GRCommandLineOptions : CommandLineOptions
    {
        /// <summary>
        /// Enables detailed logging.
        /// </summary>
        public bool DetailedLogging { get; set; } = false;

        /// <summary>
        /// Disables inspection of programmer inserted barriers.
        /// </summary>
        public bool DisableInspection { get; set; } = false;

        /// <summary>
        /// Disables grid-level barriers during instrumentation.
        /// </summary>
        public bool DisableGridBarriers { get; set; } = false;

        /// <summary>
        /// Use axioms for instrumentation instead of variable assignments.
        /// </summary>
        public bool UseAxioms { get; set; } = false;

        /// <summary>
        /// Sets the language of the input program.
        /// </summary>
        public SourceLanguage SourceLanguage { get; private set; } = SourceLanguage.OpenCL;

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
            if (name == "detailedLogging")
            {
                if (ps.ConfirmArgumentCount(1))
                    DetailedLogging = bool.Parse(ps.args[ps.i]);
                return true;
            }
            else if (name == "disableInspection")
            {
                if (ps.ConfirmArgumentCount(1))
                    DisableInspection = bool.Parse(ps.args[ps.i]);
                return true;
            }
            else if (name == "disableGridBarriers")
            {
                if (ps.ConfirmArgumentCount(1))
                    DisableGridBarriers = bool.Parse(ps.args[ps.i]);
                return true;
            }
            else if (name == "useAxioms")
            {
                if (ps.ConfirmArgumentCount(1))
                    UseAxioms = bool.Parse(ps.args[ps.i]);
                return true;
            }
            else if (name == "sourceLanguage")
            {
                if (ps.ConfirmArgumentCount(1))
                    SourceLanguage = ps.args[ps.i] == "cl" ? SourceLanguage.OpenCL : SourceLanguage.CUDA;
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
