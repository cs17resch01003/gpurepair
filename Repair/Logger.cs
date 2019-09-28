using System.IO;

namespace GPURepair.Repair
{
    public static class Logger
    {
        public static long RepairTime;

        public static int VerifierCalls;

        public static int Barriers;

        public static string ExceptionMessage;

        public static void AddRepairTime(long time)
        {
            RepairTime += time;
            VerifierCalls++;
        }

        public static void Log(string logFile, string sourceFile)
        {
            if (logFile != null)
            {
                File.AppendAllLines(logFile, new string[] { string.Format("{0},{1},{2},{3},{4}",
                    sourceFile, VerifierCalls, Barriers, RepairTime / 1000, ExceptionMessage) });
            }
        }
    }
}
