using System.Collections.Generic;
using System.IO;
using System.Text;
using GPURepair.Solvers;
using static GPURepair.Repair.Solver;

namespace GPURepair.Repair.Diagnostics
{
    public static class ClauseLogger
    {
        public static bool LogCLauses;

        public static string FileName;

        public static void Log(List<Clause> clauses, SolverType type, Dictionary<string, bool> solution)
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

                string clauseLog = string.Format("{0}.satlog", FileName);
                File.AppendAllLines(clauseLog, lines);
            }
        }
    }
}
