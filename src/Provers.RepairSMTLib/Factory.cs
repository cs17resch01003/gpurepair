namespace Microsoft.Boogie.RepairSMTLib
{
    using Microsoft.Boogie.SMTLib;

    public class Factory : SMTLib.Factory
    {
        /// <summary>
        /// Spawns a prover based on the provided options and context.
        /// </summary>
        /// <param name="options">The prover options.</param>
        /// <param name="ctxt">The context.</param>
        /// <returns>The prover.</returns>
        public override object SpawnProver(ProverOptions options, object ctxt)
        {
            SMTLibProverContext ctx = (SMTLibProverContext)ctxt;
            VCExpressionGenerator gen = ctx.ExprGen;

            return new RepairSMTLibProcessTheoremProver(options, gen, ctx);
        }
    }
}
