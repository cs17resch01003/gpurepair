using System;
using System.Diagnostics;

namespace GPURepair.Repair
{
    public class Watch : IDisposable
    {
        public static event Log LogEvent;

        public delegate void Log(long milliseconds);

        private Stopwatch watch;

        public Watch()
        {
            watch = new Stopwatch();
            watch.Start();
        }

        public void Dispose()
        {
            LogEvent?.Invoke(watch.ElapsedMilliseconds);
            watch.Stop();
        }
    }
}
