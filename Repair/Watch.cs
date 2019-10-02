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
            Logger.AddTime(((int)measure).ToString(), watch.ElapsedMilliseconds);
            watch.Stop();
        }

        public enum Measure : int
        {
            MaxSAT_SolutionTime = 1,
            MHS_SolutionTime = 2,
            VerificationTime = 3,
            OptimizationTime = 4,
        }
    }
}
