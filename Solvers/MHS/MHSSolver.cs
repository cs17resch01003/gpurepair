using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class MHSSolver
    {
        /// <summary>
        /// The clauses.
        /// </summary>
        public List<Clause> Clauses { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        public MHSSolver(List<Clause> clauses)
        {
            Clauses = clauses;
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve()
        {
            MHSSolution solution = new MHSSolution(Clauses);
            Solve(solution);

            return solution.Assignments;
        }

        /// <summary>
        /// Solves the clauses and sets the assignements.
        /// </summary>
        /// <param name="solution">The solution object that needs to be populated.</param>
        private void Solve(MHSSolution solution)
        {
            while (solution.IsSolutionFound() == false)
            {
                List<Clause> clauses = solution.GetActiveUnitClauses();
                if (clauses.Any())
                {
                    UnitLiteralPropogation(solution, clauses);
                    continue;
                }

                List<Clause> reference = solution.GetActivePositiveClauses();
                if (!reference.Any())
                    reference = solution.GetActiveNegativeClauses();

                if (reference.Any())
                    ApplyMHS(solution, reference);
            }
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
                        break;
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
            int max_unsat_clauses = 0;

            foreach (string variable in variables)
            {
                int unsat_clauses = clauses.Count(x => x.Literals.Select(y => y.Variable).Contains(variable));

                if (unsat_clauses > max_unsat_clauses)
                {
                    max_unsat_clauses = unsat_clauses;
                    chosen = variable;
                }
            }

            solution.SetAssignment(chosen, clauses.First().Literals.First().Value);
        }
    }
}