namespace GPURepair.Repair.Exceptions
{
    using System;

    public class AssertionException : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public AssertionException(string message)
            : base(message)
        {
        }
    }
}
