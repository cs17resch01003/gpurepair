namespace GPURepair.Repair.Diagnostics
{
    using System;
    using System.Diagnostics;
    using GPURepair.Common.Diagnostics;

    public class Watch : IDisposable
    {
        /// <summary>
        /// The stop watch.
        /// </summary>
        private Stopwatch watch;

        /// <summary>
        /// The code segment that is being measured.
        /// </summary>
        private Measure measure;

        /// <summary>
        /// Initializes a stop watch that logs the time taken by the given measure.
        /// </summary>
        /// <param name="measure">The code segment that is being measured.</param>
        public Watch(Measure measure)
        {
            this.measure = measure;

            watch = new Stopwatch();
            watch.Start();
        }

        /// <summary>
        /// Called when the stop watch is being disposed.
        /// </summary>
        public void Dispose()
        {
            Logger.Log($"Watch;{measure};{watch.ElapsedMilliseconds}");
            watch.Stop();
        }
    }
}
