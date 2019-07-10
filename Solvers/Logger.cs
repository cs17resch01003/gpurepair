using System;
using System.Globalization;
using System.IO;

namespace GPURepair.Solvers
{
    public static class Logger
    {
        public static string LogFile = "repair-" + DateTime.Now.ToString("yyyyMMdd-hhmmss", CultureInfo.InvariantCulture) + ".log";

        public static string Identifier;

        public static bool EnableLogging = false;

        public static void Log(string content)
        {
            if (EnableLogging)
            {
                File.AppendAllLines(LogFile, new string[] { string.Format("{0},{1},{2}", Identifier,
                    DateTime.Now.ToString("yyyyMMdd-hhmmss", CultureInfo.InvariantCulture), content) });
            }
        }
    }
}
