namespace GPURepair.Solvers.Exceptions
{
    using System;

    internal class SatisfiabilityError : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        internal SatisfiabilityError()
            : base("The given clauses are unsatisfiable!")
        {
        }
    }
}
