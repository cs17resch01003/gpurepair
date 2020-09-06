namespace GPURepair.Repair.Exceptions
{
    using System;

    public class NonBarrierException : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public NonBarrierException(string message)
            : base(message)
        {
        }
    }
}
