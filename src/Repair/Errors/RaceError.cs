namespace GPURepair.Repair.Errors
{
    using System.Collections.Generic;
    using GPURepair.Repair.Metadata;
    using GPUVerify;
    using Microsoft.Boogie;

    public class RaceError : RepairableError
    {
        /// <summary>
        /// Instantiates a new error.
        /// </summary>
        /// <param name="counterExample">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        public RaceError(Counterexample counterExample, Implementation implementation)
            : base(counterExample, implementation)
        {
        }

        /// <summary>
        /// The starting line of code which caused the race.
        /// </summary>
        public SourceLocationInfo Start { get; set; }

        /// <summary>
        /// The ending line of code which caused the race.
        /// </summary>
        public SourceLocationInfo End { get; set; }

        /// <summary>
        /// Determines if the error needs to be over-approximated or not.
        /// </summary>
        public bool Overapproximated { get; set; }

        /// <summary>
        /// The over-approximated barriers obtained from the trace when some of the barriers are in a loop.
        /// </summary>
        public IEnumerable<Barrier> OverapproximatedBarriers { get; set; }
    }
}
