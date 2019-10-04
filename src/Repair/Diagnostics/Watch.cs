using System;
using System.Diagnostics;

namespace GPURepair.Repair.Diagnostics
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
            Logger.AddTime(measure, watch.ElapsedMilliseconds);
            watch.Stop();
        }
    }
}
