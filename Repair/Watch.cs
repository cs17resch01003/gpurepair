using System;
using System.Diagnostics;

namespace GPURepair.Repair
{
    public class Watch : IDisposable
    {
        private Stopwatch watch;

        private Measure measure;

        public Watch(Measure measure)
        {
            this.measure = measure;

            watch = new Stopwatch();
            watch.Start();
        }

        public void Dispose()
        {
            Logger.AddTime(measure.ToString(), watch.ElapsedMilliseconds);
            watch.Stop();
        }

        public enum Measure
        {
            MaxSAT_SolutionTime,
            MHS_SolutionTime,
            VerificationTime,
        }
    }
}
