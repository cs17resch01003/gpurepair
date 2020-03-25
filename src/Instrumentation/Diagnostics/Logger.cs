using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GPURepair.Instrumentation.Diagnostics
{
    public static class Logger
    {
        public static string FileName;

        public static string LogFile;

        public static int CallCommands;

        public static void Flush()
        {
            if (LogFile != null)
            {
                StringBuilder builder = new StringBuilder(FileName);
                builder.Append(",");

                builder.Append(string.Join(",", CallCommands));
                File.AppendAllLines(LogFile, new List<string> { builder.ToString() });
            }
        }
    }
}
