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
    using Microsoft.Boogie;

    public class Solver
    {
        /// <summary>
        /// The weight of a grid-level barrier.
        /// </summary>
        private int gridBarrierWeight = 10;

        /// <summary>
        /// Determine the barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="type">The solver type used.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<RepairableError> errors, ref SolverType type)
        {
            List<Clause> clauses = GenerateClauses(errors);

            Dictionary<string, bool> solution;
            SolverStatus status;

            if (type == SolverType.SAT)
                solution = SolveSAT(clauses, out status);
            else if (type == SolverType.MHS)
            {
                solution = SolveMHS(clauses, out status);
                if (status != SolverStatus.Satisfiable)
                {
                    // fall back to MaxSAT if MHS fails
                    type = SolverType.MaxSAT;
                    solution = SolveMaxSAT(clauses, out status);
                }
            }
            else if (type == SolverType.MaxSAT)
                solution = SolveMaxSAT(clauses, out status);
            else
                throw new RepairError("Invalid solver type!");

            ClauseLogger.Log(clauses, type, solution);
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
        public Dictionary<string, bool> Optimize(
            List<RepairableError> errors, Dictionary<string, bool> solution, out SolverType type)
        {
            List<Clause> clauses = GenerateClauses(errors);

            Dictionary<string, int> coeffs = new Dictionary<string, int>();
            foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                coeffs.Add(barrier.Name, barrier.GridLevel ? gridBarrierWeight : 1);

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
        private List<Clause> GenerateClauses(List<RepairableError> errors)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (RepairableError error in errors)
            {
                bool value = error is RaceError;

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
        private Dictionary<Clause, uint> GenerateSoftClauses(List<string> variables)
        {
            Dictionary<Clause, uint> clauses = new Dictionary<Clause, uint>();
            foreach (string variable in variables)
            {
                Clause clause = new Clause();
                clause.Add(new Literal(variable, false));

                uint weight = (uint)(ProgramMetadata.Barriers[variable].GridLevel ? 1 : gridBarrierWeight);
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
            using (Watch watch = new Watch(Measure.MHS))
            {
                Dictionary<string, uint> weights = new Dictionary<string, uint>();
                foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                    weights.Add(barrier.Name, (uint)(barrier.GridLevel ? 1 : gridBarrierWeight));

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
            MHS,
            MaxSAT,
            Optimizer
        }
    }
}
