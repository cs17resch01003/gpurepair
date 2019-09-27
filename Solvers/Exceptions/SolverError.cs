namespace GPURepair.Solvers.Exceptions
{
    using System;

    public class SolverError : Exception
    {
        public SolverError()
            : base()
        {
        }

        public SolverError(string message)
            : base(message)
        {
        }
    }
}
