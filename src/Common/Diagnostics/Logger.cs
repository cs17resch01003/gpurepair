using System.Collections.Generic;
using System.IO;

namespace GPURepair.Common.Diagnostics
{
    public static class Logger
    {
        private static string fileName;

        private static string logFile;

        public static string FileName
        {
            get { return fileName; }
            set
            {
                fileName = value;
                logFile = new FileInfo(fileName).DirectoryName + Path.DirectorySeparatorChar + @"metrics.log";
            }
        }

        public static bool AdditionalLogging;

        public static void Log(string message)
        {
            if (AdditionalLogging)
                File.AppendAllLines(logFile, new List<string> { message });
        }
    }
}
