using System.Collections.Generic;
using Microsoft.Boogie;

namespace GPURepair.Repair.Errors
{
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
        /// The variables obtained from the trace.
        /// </summary>
        public IEnumerable<Variable> Variables { get; set; }
    }
}
