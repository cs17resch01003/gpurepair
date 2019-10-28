﻿using GPURepair.Repair.Metadata;
using Microsoft.Boogie;

namespace GPURepair.Repair.Errors
{
    public class DivergenceError : RepairableError
    {
        /// <summary>
        /// Instantiates a new error.
        /// </summary>
        /// <param name="counterExample">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        public DivergenceError(Counterexample counterExample, Implementation implementation)
            : base(counterExample, implementation)
        {
        }

        /// <summary>
        /// The line of code which caused the barrier divergence.
        /// </summary>
        public Location Location { get; set; }
    }
}