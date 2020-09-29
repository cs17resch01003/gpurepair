namespace GPURepair.Repair.Diagnostics
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using GPURepair.Repair.Metadata;
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
        /// Logs the weights of the variables.
        /// </summary>
        public static void LogWeights()
        {
            if (LogCLauses)
            {
                string message = string.Join(" ", ProgramMetadata.Barriers.Select(
                    x => string.Format("{0}[{1}]", x.Value.Name, x.Value.Weight)));

                string clauseLog = string.Format("{0}.satlog", FileName);
                File.AppendAllLines(clauseLog, new List<string> { message });
            }
        }

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

                string solution_str = string.Join(" ", solution.OrderBy(x => x.Value).ThenBy(x => int.Parse(x.Key.Replace("b", string.Empty)))
                    .Select(x => string.Format("{0}{1}", x.Value ? string.Empty : "-", x.Key)));

                List<string> clause_strings = new List<string>();
                foreach (Clause clause in clauses)
                    clause_strings.Add(clause.ToString());

                lines.Add("Solution: " + solution_str);
                lines.Add("Clauses:");
                lines.Add(string.Join(Environment.NewLine, clause_strings));

                string clauseLog = string.Format("{0}.satlog", FileName);
                File.AppendAllLines(clauseLog, lines);
            }
        }
    }
}
