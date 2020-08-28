namespace GPURepair.Repair.Exceptions
{
    using System;

    public class SummaryGeneratorError : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public SummaryGeneratorError(string message)
            : base(message)
        {
        }
    }
}
