namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Repair.Diagnostics;
    using GPURepair.Repair.Errors;
    using GPURepair.Repair.Exceptions;
    using GPURepair.Repair.Metadata;
    using GPURepair.Solvers;
    using GPURepair.Solvers.Optimizer;
    using GPURepair.Solvers.SAT;

    public class Solver
    {
        /// <summary>
        /// Determine the barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<RepairableError> errors, ref SolverType type)
        {
            try
            {
                return Solve(errors, type);
            }
            catch (RepairException)
            {
                // fall back to MaxSAT if MHS fails
                if (type != SolverType.mhs)
                    throw;

                type = SolverType.MaxSAT;
                return Solve(errors, type);
            }
        }

        /// <summary>
        /// Determine the optimized barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="solution">The solution obtained so far.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Optimize(
            List<RepairableError> errors, Dictionary<string, bool> solution, out SolverType type)
        {
            List<Clause> clauses = GenerateClauses(errors);

            Dictionary<string, int> coeffs = new Dictionary<string, int>();
            foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                coeffs.Add(barrier.Name, barrier.Weight);

            type = SolverType.Optimizer;
            while (true)
            {
                int total_weight = 0;
                foreach (string key in solution.Where(x => x.Value == true).Select(x => x.Key))
                    total_weight += coeffs[key];

                using (Watch watch = new Watch(Measure.Optimization))
                {
                    SolverStatus status;

                    Optimizer optimizer = new Optimizer(clauses, coeffs);
                    Dictionary<string, bool> assignments = optimizer.Solve(--total_weight, out status);

                    if (status == SolverStatus.Unsatisfiable)
                    {
                        ClauseLogger.Log(clauses, type, solution);
                        return solution;
                    }
                    else
                        solution = assignments;
                }
            }
        }

        private Dictionary<string, bool> Solve(List<RepairableError> errors, SolverType type)
        {
            try
            {
                List<Clause> clauses = GenerateClauses(errors);
                return Solve(clauses, type);
            }

            // the repair exception could be because of not considering all the barriers in the loop
            // refer CUDA/transitiveclosure as an example
            catch (RepairException)
            {
                bool overapproximated = false;
                foreach (Error error in errors)
                {
                    RaceError race = error as RaceError;
                    if (race != null && race.OverapproximatedBarriers.Any())
                    {
                        overapproximated = true;
                        race.Overapproximated = true;
                    }
                }

                // if none of the clauses were overapproximated, return the original result
                if (!overapproximated)
                    throw;

                List<Clause> clauses = GenerateClauses(errors);
                return Solve(clauses, type);
            }
        }

        private Dictionary<string, bool> Solve(List<Clause> clauses, SolverType type)
        {
            Dictionary<string, bool> solution;
            SolverStatus status;

            if (type == SolverType.SAT)
                solution = SolveSAT(clauses, out status);
            else if (type == SolverType.mhs)
                solution = SolveMHS(clauses, out status);
            else if (type == SolverType.MaxSAT)
                solution = SolveMaxSAT(clauses, out status);
            else
                throw new RepairException("Invalid solver type!");

            ClauseLogger.Log(clauses, type, solution);
            if (status == SolverStatus.Unsatisfiable)
                throw new RepairException("The program could not be repaired because of unsatisfiable clauses!");

            return solution;
        }

        /// <summary>
        /// Generates clauses based on the errors.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The clauses.</returns>
        private List<Clause> GenerateClauses(List<RepairableError> errors)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (RepairableError error in errors)
            {
                RaceError race = error as RaceError;
                if (race != null)
                {
                    IEnumerable<Barrier> barriers = error.Barriers;
                    if (race != null && race.Overapproximated && race.OverapproximatedBarriers.Any())
                    {
                        // consider the over-approximation barriers for the loop
                        barriers = race.OverapproximatedBarriers;
                    }

                    Clause clause = new Clause();
                    foreach (string variable in barriers.Select(x => x.Name))
                        clause.Add(new Literal(variable, true));

                    clauses.Add(clause);
                }
                else
                {
                    foreach (string variable in error.Barriers.Select(x => x.Name))
                    {
                        Clause clause = new Clause();
                        clause.Add(new Literal(variable, false));

                        clauses.Add(clause);
                    }
                }
            }

            return clauses.Distinct().ToList();
        }

        /// <summary>
        /// Generates soft clauses based on the variables.
        /// </summary>
        /// <param name="variables">The variables.</param>
        /// <returns>The soft clauses.</returns>
        private Dictionary<Clause, uint> GenerateSoftClauses(List<string> variables)
        {
            Dictionary<Clause, uint> clauses = new Dictionary<Clause, uint>();
            foreach (string variable in variables)
            {
                Clause clause = new Clause();
                clause.Add(new Literal(variable, false));

                uint weight = (uint)ProgramMetadata.Barriers[variable].Weight;
                clauses.Add(clause, weight);
            }

            return clauses;
        }

        /// <summary>
        /// Solves the clauses using a SAT solver.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        private Dictionary<string, bool> SolveSAT(List<Clause> clauses, out SolverStatus status)
        {
            using (Watch watch = new Watch(Measure.SAT))
            {
                SATSolver solver = new SATSolver(clauses);
                return solver.Solve(out status);
            }
        }

        /// <summary>
        /// Solves the clauses using a MHS solver.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        private Dictionary<string, bool> SolveMHS(List<Clause> clauses, out SolverStatus status)
        {
            using (Watch watch = new Watch(Measure.mhs))
            {
                Dictionary<string, double> weights = new Dictionary<string, double>();
                foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                    weights.Add(barrier.Name, 1.0 / barrier.Weight);

                MHSSolver solver = new MHSSolver(clauses, weights);
                return solver.Solve(out status);
            }
        }

        /// <summary>
        /// Solves the clauses using a MaxSAT solver.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        private Dictionary<string, bool> SolveMaxSAT(List<Clause> clauses, out SolverStatus status)
        {
            using (Watch watch = new Watch(Measure.MaxSAT))
            {
                List<string> variables = clauses.SelectMany(x => x.Literals)
                    .Select(x => x.Variable).Distinct().ToList();
                Dictionary<Clause, uint> soft_clauses = GenerateSoftClauses(variables);

                MaxSATSolver solver = new MaxSATSolver(clauses, soft_clauses);
                return solver.Solve(out status);
            }
        }

        /// <summary>
        /// The solver type used.
        /// </summary>
        public enum SolverType
        {
            SAT,
            mhs,
            MaxSAT,
            Optimizer
        }
    }
}
