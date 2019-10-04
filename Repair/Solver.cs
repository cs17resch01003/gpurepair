using GPURepair.Repair.Diagnostics;
using GPURepair.Repair.Exceptions;
using GPURepair.Solvers;
using GPURepair.Solvers.Optimizer;
using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Repair
{
    public class Solver
    {
        /// <summary>
        /// Determine the barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<Error> errors, out SolverType type)
        {
            List<Clause> clauses = GenerateClauses(errors);

            Dictionary<string, bool> solution;
            SolverStatus status;

            type = SolverType.MHS;
            using (Watch watch = new Watch(Measure.MHS))
            {
                MHSSolver solver = new MHSSolver(clauses);
                solution = solver.Solve(out status);
            }

            if (status != SolverStatus.Satisfiable)
            {
                // fall back to MaxSAT if MHS fails
                type = SolverType.MaxSAT;

                using (Watch watch = new Watch(Measure.MaxSAT))
                {
                    List<string> variables = clauses.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct().ToList();
                    List<Clause> soft_clauses = GenerateSoftClauses(variables);

                    MaxSATSolver solver = new MaxSATSolver(clauses, soft_clauses);
                    solution = solver.Solve(out status);
                }
            }

            Logger.LogClausesToFile(clauses, type, solution);
            if (status == SolverStatus.Unsatisfiable)
                throw new RepairError("The program could not be repaired because of unsatisfiable clauses!");

            return solution;
        }

        /// <summary>
        /// Determine the optimized barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="solution">The solution obtained so far.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Optimize(List<Error> errors, Dictionary<string, bool> solution, out SolverType type)
        {
            int barrier_count = solution.Count(x => x.Value == true);
            List<Clause> clauses = GenerateClauses(errors);

            type = SolverType.Optimizer;
            while (true)
            {
                using (Watch watch = new Watch(Measure.Optimization))
                {
                    SolverStatus status;

                    Optimizer optimizer = new Optimizer(clauses);
                    Dictionary<string, bool> assignments = optimizer.Solve(--barrier_count, out status);

                    if (status == SolverStatus.Unsatisfiable)
                        return solution;
                    else
                        solution = assignments;
                }
            }
        }

        /// <summary>
        /// Generates clauses based on the errors.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The clauses.</returns>
        private List<Clause> GenerateClauses(List<Error> errors)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (Error error in errors)
            {
                bool value = error.ErrorType == ErrorType.Race;

                Clause clause = new Clause();
                foreach (Variable variable in error.Variables)
                    clause.Add(new Literal(variable.Name, value));

                clauses.Add(clause);
            }

            return clauses;
        }

        /// <summary>
        /// Generates soft clauses based on the variables.
        /// </summary>
        /// <param name="variables">The variables.</param>
        /// <returns>The soft clauses.</returns>
        private List<Clause> GenerateSoftClauses(List<string> variables)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (string variable in variables)
            {
                Clause clause = new Clause();
                clause.Add(new Literal(variable, false));

                clauses.Add(clause);
            }

            return clauses;
        }

        /// <summary>
        /// The solver type used.
        /// </summary>
        public enum SolverType
        {
            MHS,
            MaxSAT,
            Optimizer
        }
    }
}