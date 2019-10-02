using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace GPURepair.Repair.Diagnostics
{
    public static class Logger
    {
        private static Dictionary<Measure, double> time = new Dictionary<Measure, double>();

        private static Dictionary<Measure, int> count = new Dictionary<Measure, int>();

        public static int? Barriers;

        public static int? Changes;

        public static string ExceptionMessage;

        public static void AddTime(Measure measure, long time)
        {
            if (!Logger.time.ContainsKey(measure))
                Logger.time[measure] = 0;

            if (!count.ContainsKey(measure))
                count[measure] = 0;

            Logger.time[measure] += time;
            count[measure]++;
        }

        public static void Log(string logFile, string sourceFile)
        {
            if (logFile != null)
            {
                StringBuilder builder = new StringBuilder(sourceFile);
                builder.Append(",");

                foreach (Measure measure in Enum.GetValues(typeof(Measure)))
                {
                    double time = Logger.time.ContainsKey(measure) ? Logger.time[measure] / 1000.0 : 0;
                    int count = Logger.count.ContainsKey(measure) ? Logger.count[measure] : 0;

                    builder.Append(string.Format("{0},{1},", time, count));
                }

                builder.Append(string.Join(",", Barriers, Changes, ExceptionMessage));
                File.AppendAllLines(logFile, new List<string> { builder.ToString() });
            }
        }
    }
}
