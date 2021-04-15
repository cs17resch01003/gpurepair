namespace GPURepair.Repair.Verifiers
{
    using System;
    using System.Collections.Generic;
    using GPURepair.Repair.Boogie;
    using Microsoft.Boogie;

    public class IncrementalVerifier : Verifier, IDisposable
    {
        /// <summary>
        /// The extended Boogie verifier.
        /// </summary>
        private RepairVCGen verifier;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public IncrementalVerifier(Program program)
            : base(program)
        {
            verifier = new RepairVCGen(
                program,
                CommandLineOptions.Clo.SimplifyLogFilePath,
                CommandLineOptions.Clo.SimplifyLogFileAppend);
        }

        /// <summary>
        /// Gets the errors and associated metadata after verifying the program.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The counter examples.</returns>
        protected override IEnumerable<Counterexample> VerifyImplementation(
            Implementation implementation, Dictionary<string, bool> assignments)
        {
            return verifier.VerifyImplementation(implementation, assignments);
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
