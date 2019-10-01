namespace GPURepair.Solvers.Exceptions
{
    using System;

    public class SatisfiabilityError : Exception
    {
        public SatisfiabilityError()
            : base("The given clauses are unsatisfiable!")
        {
        }
    }
}