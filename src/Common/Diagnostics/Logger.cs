namespace GPURepair.Common.Diagnostics
{
    using System.Collections.Generic;
    using System.IO;

    public static class Logger
    {
        /// <summary>
        /// The input filename.
        /// </summary>
        private static string fileName;

        /// <summary>
        /// The log file name.
        /// </summary>
        private static string logFile;

        /// <summary>
        /// Gets or sets the input filename.
        /// </summary>
        public static string FileName
        {
            get { return fileName; }
            set
            {
                fileName = value;
                logFile = new FileInfo(fileName).DirectoryName + Path.DirectorySeparatorChar + @"metrics.log";
            }
        }

        /// <summary>
        /// Enables detailed logging.
        /// </summary>
        public static bool DetailedLogging;

        /// <summary>
        /// Logs the message in the log file.
        /// </summary>
        /// <param name="message">The message.</param>
        public static void Log(string message)
        {
            if (DetailedLogging)
                File.AppendAllLines(logFile, new List<string> { message });
        }
    }
}
