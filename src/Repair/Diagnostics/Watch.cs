using System;
using System.Diagnostics;
using GPURepair.Common.Diagnostics;

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
            Logger.Log($"Watch;{measure};{watch.ElapsedMilliseconds}");
            watch.Stop();
        }
    }
}
