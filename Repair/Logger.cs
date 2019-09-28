using System.IO;

namespace GPURepair.Repair
{
    public static class Logger
    {
        public static double VerificationTime;

        public static int? VerifierCalls;

        public static int? Barriers;

        public static int? Changes;

        public static string ExceptionMessage;

        public static void AddVerificationTime(long time)
        {
            VerificationTime += time;
            VerifierCalls++;
        }

        public static void Log(string logFile, string sourceFile)
        {
            if (logFile != null)
            {
                File.AppendAllLines(logFile, new string[] { string.Format("{0},{1},{2},{3},{4},{5}",
                    sourceFile, VerifierCalls, Barriers, Changes, VerificationTime / 1000.0, ExceptionMessage) });
            }
        }
    }
}
