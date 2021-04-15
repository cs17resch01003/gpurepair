namespace GPURepair.Repair.Verifiers
{
    using System;
    using System.Collections.Generic;
    using Microsoft.Boogie;
    using VC;

    public class ClassicVerifier : Verifier, IDisposable
    {
        /// <summary>
        /// The Boogie verifier.
        /// </summary>
        private VCGen verifier;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public ClassicVerifier(Program program)
            : base(program)
        {
            verifier = new VCGen(
                program,
                CommandLineOptions.Clo.SimplifyLogFilePath,
                CommandLineOptions.Clo.SimplifyLogFileAppend,
                new List<Checker>());
        }

        /// <summary>
        /// Gets the counter examples after verifying the program.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The counter examples.</returns>
        protected override IEnumerable<Counterexample> VerifyImplementation(
            Implementation implementation, Dictionary<string, bool> assignments)
        {
            List<Counterexample> examples;
            ConditionGeneration.Outcome outcome = verifier.VerifyImplementation(implementation, out examples);
            if (outcome == ConditionGeneration.Outcome.Errors)
                return examples;

            return new List<Counterexample>();
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            verifier.Close();
        }
    }
}
