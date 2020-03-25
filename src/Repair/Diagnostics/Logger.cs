using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using GPURepair.Solvers;
using static GPURepair.Repair.Solver;

namespace GPURepair.Repair.Diagnostics
{
    public static class Logger
    {
        private static Dictionary<Measure, double> time = new Dictionary<Measure, double>();

        private static Dictionary<Measure, int> count = new Dictionary<Measure, int>();

        public static bool LogCLauses;

        public static string FileName;

        public static string LogFile;

        public static int? Barriers;

        public static int? Changes;

        public static int VerifierRunsAfterOptimization;

        public static int VerifierFailuresAfterOptimization;

        public static int? BarriersBetweenCalls;

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

        public static void Flush()
        {
            if (LogFile != null)
            {
                StringBuilder builder = new StringBuilder(FileName);
                builder.Append(",");

                foreach (Measure measure in Enum.GetValues(typeof(Measure)))
                {
                    double time = Logger.time.ContainsKey(measure) ? Logger.time[measure] / 1000.0 : 0;
                    int count = Logger.count.ContainsKey(measure) ? Logger.count[measure] : 0;

                    builder.Append(string.Format("{0},{1},", time, count));
                }

                builder.Append(string.Join(",",
                    Barriers, Changes, VerifierRunsAfterOptimization,
                    VerifierFailuresAfterOptimization, BarriersBetweenCalls, ExceptionMessage));
                File.AppendAllLines(LogFile, new List<string> { builder.ToString() });
            }
        }

        public static void LogClausesToFile(List<Clause> clauses, SolverType type, Dictionary<string, bool> solution)
        {
            if (LogCLauses)
            {
                List<string> lines = new List<string>();
                lines.Add("Type: " + type.ToString());

                StringBuilder builder = new StringBuilder();
                foreach (KeyValuePair<string, bool> pair in solution)
                    builder.AppendFormat("{0}{1} ", pair.Value ? string.Empty : "-", pair.Key);

                lines.Add("Solution: " + builder.ToString());

                lines.Add("Clauses:");
                foreach (Clause clause in clauses)
                    lines.Add(clause.ToString());

                string clauseLog = string.Format("{0}_{1}.satlog", FileName, DateTime.Now.ToString("hhmmss"));
                File.WriteAllLines(clauseLog, lines);
            }
        }
    }
}
