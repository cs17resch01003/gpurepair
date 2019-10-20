using System.Collections.Generic;
using Microsoft.Boogie;

namespace GPURepair.Repair
{
    public class Error
    {
        /// <summary>
        /// The counter example associated with the error.
        /// </summary>
        public Counterexample CounterExample { get; set; }

        /// <summary>
        /// Indicates whether the error is a race or a divergence.
        /// </summary>
        public ErrorType? ErrorType { get; set; }

        /// <summary>
        /// The implementation where the error originated.
        /// </summary>
        public Implementation Implementation { get; set; }

        /// <summary>
        /// The variables obtained from the trace.
        /// </summary>
        public IEnumerable<Variable> Variables { get; set; }
    }
}
