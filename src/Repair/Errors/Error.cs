namespace GPURepair.Repair.Errors
{
    using Microsoft.Boogie;

    public class Error
    {
        /// <summary>
        /// Instantiates a new error.
        /// </summary>
        /// <param name="counterExample">The counter example.</param>
        /// <param name="implementation">The implementation.</param>
        public Error(Counterexample counterExample, Implementation implementation)
        {
            CounterExample = counterExample;
            Implementation = implementation;
        }

        /// <summary>
        /// The counter example associated with the error.
        /// </summary>
        public Counterexample CounterExample { get; set; }

        /// <summary>
        /// The implementation where the error originated.
        /// </summary>
        public Implementation Implementation { get; set; }
    }
}
