using System;
using System.Globalization;
using System.IO;

namespace GPURepair.Solvers
{
    public static class Logger
    {
        public static string LogFile;

        public static string Identifier;

        public static void Log(string content)
        {
            if (LogFile != null)
            {
                File.AppendAllLines(LogFile, new string[] { string.Format("{0},{1},{2}", Identifier,
                    DateTime.Now.ToString("yyyyMMdd-hhmmss", CultureInfo.InvariantCulture), content) });
            }
        }
    }
}
