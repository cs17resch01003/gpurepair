namespace GPURepair.Repair.Exceptions
{
    using System;

    public class SummaryGeneratorException : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public SummaryGeneratorException(string message)
            : base(message)
        {
        }
    }
}
