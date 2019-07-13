using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GPURepair.Solvers
{
    public static class Logger
    {
        private static List<long> repairTimes = new List<long>();

        private static List<bool> solverStatuses = new List<bool>();

        public static int Barriers;

        public static void LogRepairTime(long time)
        {
            repairTimes.Add(time);
        }

        public static void LogSolverStatus(bool status)
        {
            solverStatuses.Add(status);
        }

        public static void Log(string logFile, string sourceFile)
        {
            if (logFile != null)
            {
                File.AppendAllLines(logFile, new string[] { string.Format("{0},{1},{2},{3},{4},{5}", sourceFile, Barriers,
                    solverStatuses.Count(x => x == true), solverStatuses.Count(x => x == false), repairTimes.Count,
                    string.Join(",", repairTimes)) });
            }
        }
    }
}
