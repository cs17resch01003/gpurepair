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
            Logger.AddVerificationTime(watch.ElapsedMilliseconds);
            watch.Stop();
        }
    }
}
