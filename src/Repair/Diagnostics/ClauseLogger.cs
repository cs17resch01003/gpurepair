namespace GPURepair.Repair.Diagnostics
{
    using System.Collections.Generic;
    using System.IO;
    using System.Text;
    using GPURepair.Solvers;
    using static GPURepair.Repair.Solver;

    public static class ClauseLogger
    {
        /// <summary>
        /// Enables logging the clauses to the file.
        /// </summary>
        public static bool LogCLauses;

        /// <summary>
        /// The file used for logging.
        /// </summary>
        public static string FileName;

        /// <summary>
        /// Logs the clauses and the solution to the log file.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="type">The solver type.</param>
        /// <param name="solution">The solution.</param>
        public static void Log(List<Clause> clauses, SolverType type, Dictionary<string, bool> solution)
        {
            if (LogCLauses)
            {
                List<string> lines = new List<string>();
                lines.Add("Type: " + type.ToString());

                StringBuilder builder = new StringBuilder();
                if (solution != null)
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
