namespace GPURepair.Repair.Boogie
{
    using System.Collections.Generic;
    using GPURepair.Repair.Metadata;
    using Microsoft.Boogie;
    using Microsoft.Boogie.RepairSMTLib;
    using VC;

    public class RepairVCGen : VCGen
    {
        /// <summary>
        /// A cache to store the mappings between the implementations and the checkers.
        /// </summary>
        private Dictionary<Implementation, ImplementationCacheValue> cache =
            new Dictionary<Implementation, ImplementationCacheValue>();

        /// <summary>
        /// Creates an instance of the repair-based VCGen.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        /// <param name="logFilePath">The path of the log file.</param>
        /// <param name="appendLogFile">Appends to the log file if true.</param>
        public RepairVCGen(Program program, string logFilePath, bool appendLogFile)
            : base(program, logFilePath, appendLogFile, new List<Checker>())
        {
        }

        /// <summary>
        /// Verifies the provided implementation.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        /// <returns>The counter examples.</returns>
        public IEnumerable<Counterexample> VerifyImplementation(Implementation implementation,
            Dictionary<string, bool> assignments)
        {
            List<Counterexample> examples = new List<Counterexample>();
            if (!cache.ContainsKey(implementation))
            {
                CounterexampleCollector collector = new CounterexampleCollector();
                List<Split> splits;

                Outcome outcome = VerifyImplementation(implementation, collector, out splits);
                if (outcome == Outcome.Errors)
                    examples.AddRange(collector.examples);

                cache.Add(implementation, new ImplementationCacheValue
                {
                    Collector = collector,
                    Splits = splits,
                });
            }
            else
            {
                ImplementationCacheValue cacheValue = cache[implementation];
                foreach (Split split in cacheValue.Splits)
                {
                    cacheValue.Collector.examples.Clear();
                    bool prover_failed;

                    RepairSMTLibProcessTheoremProver prover =
                        split.Checker.TheoremProver as RepairSMTLibProcessTheoremProver;
                    prover.Send("(pop 2)");
                    prover.Send("(push 2)");

                    ConstraintProgram(implementation, prover, assignments);
                    prover.SendCheckSat();

                    Outcome outcome = Outcome.Correct;
                    split.Checker.WaitForOutput(null);

                    split.ReadOutcome(ref outcome, out prover_failed);
                    if (outcome == Outcome.Errors)
                        examples.AddRange(cacheValue.Collector.examples);
                }
            }

            return examples;
        }

        /// <summary>
        /// Retrieves a checker for the split.
        /// </summary>
        /// <param name="timeout">The timeout.</param>
        /// <param name="rlimit">The resource limit.</param>
        /// <param name="isBlocking">Waits for the give time if this is set to true.</param>
        /// <param name="waitTimeinMs">The wait time.</param>
        /// <param name="maxRetries">The maximum of retries.</param>
        /// <returns>The checker.</returns>
        protected override Checker FindCheckerFor(
            int timeout, int rlimit = 0, bool isBlocking = true, int waitTimeinMs = 50, int maxRetries = 3)
        {
            string log = logFilePath;
            if (log != null && !log.Contains("@PROC@") && checkers.Count > 0)
            {
                log = log + "." + checkers.Count;
            }

            Checker checker = new Checker(this, program, log, appendLogFile, timeout, rlimit);
            checker.GetReady();

            checkers.Add(checker);
            return checker;
        }

        /// <summary>
        /// Sends SMTLib instructions to the prover based on the provided assignments.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="prover">The theorem prover.</param>
        /// <param name="assignments">The current assignments to the variables.</param>
        private void ConstraintProgram(Implementation implementation,
            RepairSMTLibProcessTheoremProver prover,
            Dictionary<string, bool> assignments)
        {
            string format;
            foreach (KeyValuePair<string, bool> assignment in assignments)
            {
                Barrier barrier = ProgramMetadata.Barriers[assignment.Key];
                if (barrier.Implementation.Name == implementation.Name)
                {
                    format = assignment.Value ? "(assert {0})" : "(assert (not {0}))";
                    prover.Send(string.Format(format, assignment.Key));
                }
            }
        }

        /// <summary>
        /// Represents a value in the implementation cache.
        /// </summary>
        private class ImplementationCacheValue
        {
            /// <summary>
            /// The counter-example collector.
            /// </summary>
            public CounterexampleCollector Collector { get; set; }

            /// <summary>
            /// The splits obtained from the input program.
            /// </summary>
            public IEnumerable<Split> Splits { get; set; }
        }
    }
}
