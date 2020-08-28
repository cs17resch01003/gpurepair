namespace GPURepair.Repair.Exceptions
{
    using System;

    public class NonBarrierError : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public NonBarrierError(string message)
            : base(message)
        {
        }
    }
}
