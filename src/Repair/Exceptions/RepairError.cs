namespace GPURepair.Repair.Exceptions
{
    using System;

    public class RepairError : Exception
    {
        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="message">The error message.</param>
        public RepairError(string message)
            : base(message)
        {
        }
    }
}
