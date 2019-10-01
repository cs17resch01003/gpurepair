using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace GPURepair.Repair
{
    public static class Logger
    {
        private static Dictionary<string, double> time = new Dictionary<string, double>();

        private static Dictionary<string, int> count = new Dictionary<string, int>();

        public static int? Barriers;

        public static int? Changes;

        public static string ExceptionMessage;

        public static void AddTime(string property, long time)
        {
            if (!Logger.time.ContainsKey(property))
                Logger.time[property] = 0;

            if (!count.ContainsKey(property))
                count[property] = 0;

            Logger.time[property] += time;
            count[property]++;
        }

        public static void Log(string logFile, string sourceFile)
        {
            if (logFile != null)
            {
                StringBuilder builder = new StringBuilder(sourceFile);
                builder.Append(",");

                foreach (string property in time.Keys.OrderBy(x => x))
                    builder.Append(string.Format("{0},{1},", time[property] / 1000.0, count[property]));
                builder.Append(string.Join(",", Barriers, Changes, ExceptionMessage));
            }
        }
    }
}
