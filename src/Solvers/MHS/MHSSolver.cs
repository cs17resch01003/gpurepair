namespace GPURepair.Solvers
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Solvers.Exceptions;

    public class MHSSolver
    {
        /// <summary>
        /// The clauses.
        /// </summary>
        private List<Clause> clauses { get; set; }

        /// <summary>
        /// The weights associated with the literals.
        /// </summary>
        private Dictionary<string, uint> weights { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="weights">The weights associated with the literals.</param>
        public MHSSolver(List<Clause> clauses, Dictionary<string, uint> weights)
        {
            this.clauses = clauses;
            this.weights = weights;
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve(out SolverStatus status)
        {
            try
            {
                MHSSolution solution = new MHSSolution(clauses);
                Solve(solution);

                status = SolverStatus.Satisfiable;
                return solution.Assignments;
            }
            catch (SatisfiabilityError)
            {
                status = SolverStatus.Unsatisfiable;
                return null;
            }
        }

        /// <summary>
        /// Solves the clauses and sets the assignements.
        /// </summary>
        /// <param name="solution">The solution object that needs to be populated.</param>
        private void Solve(MHSSolution solution)
        {
            // unit literal propogation
            List<Clause> unit_clauses = solution.GetActiveUnitClauses();
            while (unit_clauses.Any())
            {
                UnitLiteralPropogation(solution, unit_clauses);
                unit_clauses = solution.GetActiveUnitClauses();
            }

            // find MHS for all positive clauses
            List<Clause> positive_clauses = solution.GetActivePositiveClauses();
            while (positive_clauses.Any())
            {
                ApplyMHS(solution, positive_clauses);
                positive_clauses = solution.GetActivePositiveClauses();
            }

            // set everything else to false
            foreach (string variable in solution.VariableLookup.Keys)
                if (!solution.Assignments.ContainsKey(variable))
                    solution.SetAssignment(variable, false);
        }

        /// <summary>
        /// Propogates the unit literals to the other clauses.
        /// </summary>
        /// <param name="solution">The solution object that needs to be populated.</param>
        /// <param name="clauses">The clauses.</param>
        private void UnitLiteralPropogation(MHSSolution solution, List<Clause> clauses)
        {
            foreach (Clause clause in clauses)
            {
                foreach (Literal literal in clause.Literals)
                    if (!solution.Assignments.ContainsKey(literal.Variable))
                    {
                        solution.SetAssignment(literal.Variable, literal.Value);
                        return;
                    }
            }
        }

        /// <summary>
        /// Identifies the assignment that satisfies maximum number of clauses.
        /// </summary>
        /// <param name="solution">The solution object that needs to be populated.</param>
        /// <param name="clauses">The clauses.</param>
        private void ApplyMHS(MHSSolution solution, List<Clause> clauses)
        {
            IEnumerable<string> variables = clauses.SelectMany(x => x.Literals)
                .Select(x => x.Variable).Distinct();

            string chosen = string.Empty;
            long max_total_weight = 0;

            foreach (string variable in variables)
            {
                if (!solution.Assignments.ContainsKey(variable))
                {
                    int unknown_clauses = clauses.Count(x => x.Literals.Select(y => y.Variable).Contains(variable));
                    long total_weight = unknown_clauses * weights[variable];

                    if (total_weight > max_total_weight)
                    {
                        max_total_weight = total_weight;
                        chosen = variable;
                    }
                }
            }

            solution.SetAssignment(chosen, clauses.First().Literals.First().Value);
        }
    }
}
