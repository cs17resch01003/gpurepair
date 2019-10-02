using GPURepair.Solvers;
using GPURepair.Solvers.Exceptions;
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
            try
            {
                using (Watch watch = new Watch(Watch.Measure.MHS_SolutionTime))
                {
                    List<Clause> clauses = GenerateClauses(errors);
                    MHSSolver solver = new MHSSolver(clauses);

                    type = SolverType.MHS;
                    return solver.Solve();
                }
            }
            catch (SatisfiabilityError)
            {
                using (Watch watch = new Watch(Watch.Measure.MaxSAT_SolutionTime))
                {
                    // fall back to MaxSAT if MHS fails
                    List<Clause> clauses = GenerateClauses(errors);
                    List<string> variables = clauses.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct().ToList();

                    List<Clause> soft_clauses = GenerateSoftClauses(variables);
                    MaxSATSolver solver = new MaxSATSolver(clauses, soft_clauses);

                    type = SolverType.MaxSAT;
                    return solver.Solve();
                }
            }
        }

        /// <summary>
        /// Determine the optimized barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="solution">The solution obtained so far.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Optimize(List<Error> errors, Dictionary<string, bool> solution,
            out SolverType type)
        {
            int barrier_count = solution.Count(x => x.Value == true);
            List<Clause> clauses = GenerateClauses(errors);

            while (true)
            {
                using (Watch watch = new Watch(Watch.Measure.OptimizationTime))
                {
                    try
                    {
                        Optimizer optimizer = new Optimizer(clauses);
                        solution = optimizer.Solve(--barrier_count);
                    }
                    catch (SatisfiabilityError)
                    {
                        type = SolverType.Optimizer;
                        return solution;
                    }
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