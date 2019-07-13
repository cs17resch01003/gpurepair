using System;
using System.Diagnostics;
using GPURepair.Solvers;

namespace GPURepair.Repair
{
    public class Watch : IDisposable
    {
        private Stopwatch watch;

        public Watch()
        {
            watch = new Stopwatch();
            watch.Start();
        }

        public void Dispose()
        {
            Logger.LogRepairTime(watch.ElapsedMilliseconds);
            watch.Stop();
        }
    }
}
