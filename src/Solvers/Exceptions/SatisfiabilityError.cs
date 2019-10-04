namespace GPURepair.Solvers.Exceptions
{
    using System;

    internal class SatisfiabilityError : Exception
    {
        internal SatisfiabilityError()
            : base("The given clauses are unsatisfiable!")
        {
        }
    }
}
