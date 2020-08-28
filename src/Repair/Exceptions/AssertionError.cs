namespace GPURepair.Repair.Exceptions
{
    using System;

    public class AssertionError : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public AssertionError(string message)
            : base(message)
        {
        }
    }
}
