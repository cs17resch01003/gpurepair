namespace GPURepair.Repair.Exceptions
{
    using System;

    public class RepairException : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public RepairException(string message)
            : base(message)
        {
        }
    }
}
