namespace GPURepair.Repair.Errors
{
    using System.Collections.Generic;
    using GPURepair.Repair.Metadata;
    using Microsoft.Boogie;

    public abstract class RepairableError : Error
    {
        /// <summary>
        /// Instantiates a new error.
        /// </summary>
        /// <param name="counterExample">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        public RepairableError(Counterexample counterExample, Implementation implementation)
            : base(counterExample, implementation)
        {
        }

        /// <summary>
        /// The barriers obtained from the trace.
        /// </summary>
        public IEnumerable<Barrier> Barriers { get; set; }
    }
}
