namespace GPURepair.Instrumentation
{
    using GPURepair.Common;
    using Microsoft.Boogie;

    public class GRCommandLineOptions : CommandLineOptions
    {
        /// <summary>
        /// Enables additional logging.
        /// </summary>
        public bool AdditionalLogging { get; set; } = false;

        /// <summary>
        /// Enables instrumentation of programmer inserted barriers.
        /// </summary>
        public bool InstrumentExistingBarriers { get; set; } = true;

        /// <summary>
        /// Enables grid-level barriers during instrumentation.
        /// </summary>
        public bool EnableGridBarriers { get; set; } = true;

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
            if (name == "additionalLogging")
            {
                if (ps.ConfirmArgumentCount(1))
                    AdditionalLogging = bool.Parse(ps.args[ps.i]);
                return true;
            }

            if (name == "instrumentExistingBarriers")
            {
                if (ps.ConfirmArgumentCount(1))
                    InstrumentExistingBarriers = bool.Parse(ps.args[ps.i]);
                return true;
            }

            if (name == "enableGridBarriers")
            {
                if (ps.ConfirmArgumentCount(1))
                    EnableGridBarriers = bool.Parse(ps.args[ps.i]);
                return true;
            }

            if (name == "sourceLanguage")
            {
                if (ps.ConfirmArgumentCount(1))
                    SourceLanguage = ps.args[ps.i] == "cl" ? SourceLanguage.OpenCL : SourceLanguage.CUDA;
                return true;
            }

            return base.ParseOption(name, ps);
        }
    }
}
