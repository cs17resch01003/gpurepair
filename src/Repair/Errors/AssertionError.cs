namespace GPURepair.Repair.Errors
{
    using Microsoft.Boogie;

    public class AssertionError : RepairableError
    {
        /// <summary>
        /// Instantiates a new error.
        /// </summary>
        /// <param name="counterExample">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        public AssertionError(Counterexample counterExample, Implementation implementation)
            : base(counterExample, implementation)
        {
        }
    }
}
