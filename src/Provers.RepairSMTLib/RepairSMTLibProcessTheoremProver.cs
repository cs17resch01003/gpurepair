namespace Microsoft.Boogie.RepairSMTLib
{
    using Microsoft.Boogie.SMTLib;
    using Microsoft.Boogie.VCExprAST;

    public class RepairSMTLibProcessTheoremProver : SMTLibProcessTheoremProver
    {
        /// <summary>
        /// Creates a new instance of the repair-based theorem prover.
        /// </summary>
        /// <param name="options">The prover options.</param>
        /// <param name="gen">The expression generator.</param>
        /// <param name="ctx">The prover context.</param>
        public RepairSMTLibProcessTheoremProver(
            ProverOptions options,
            VCExpressionGenerator gen,
            SMTLibProverContext ctx) : base(options, gen, ctx)
        {
        }

        /// <summary>
        /// Initializes the verification process.
        /// </summary>
        /// <param name="descriptiveName">The name of the check.</param>
        /// <param name="vc">The expression.</param>
        /// <param name="handler">The error handler.</param>
        public override void BeginCheck(string descriptiveName, VCExpr vc, ErrorHandler handler)
        {
            BeginCheckInternal(descriptiveName, vc, handler);
            SendThisVC("(push 2)");
            SendCheckSat();
            FlushLogFile();
        }

        /// <summary>
        /// Returns the outcome of the verification process.
        /// </summary>
        /// <param name="handler">he error handler.</param>
        /// <param name="taskID">The task id.</param>
        /// <returns>The outcome of the verification process.</returns>
        public override Outcome CheckOutcome(ErrorHandler handler, int taskID = -1)
        {
            Outcome outcome = base.CheckOutcomeCore(handler, taskID);
            return outcome;
        }

        /// <summary>
        /// Sends the provided SMTLib instruction to the solver.
        /// </summary>
        /// <param name="s">The SMTLib instruction.</param>
        public void Send(string s)
        {
            SendThisVC(s);
        }
    }
}
