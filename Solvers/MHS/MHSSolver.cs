using System;
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
            bool result = Solve(solution);

            // Use the SATSolver as a backup if the MHSSolver fails
            if (result == true)
            {
                Logger.Log("MHSSolver Succeeded!");
                return solution.Assignments;
            }
            else
            {
                SATSolver solver = new SATSolver(Clauses);
                Dictionary<string, bool> assignments = solver.Solve();

                if (assignments == null)
                    throw new Exception("The given clauses cannot be satisfied!");

                Logger.Log("MHSSolver Failed!");
                return assignments;
            }
        }

        /// <summary>
        /// Solves the clauses and sets the assignements.
        /// </summary>
        /// <param name="solution">The solution object that needs to be populated.</param>
        /// <returns>true if a solution has been found and false otherwise.</returns>
        private bool Solve(MHSSolution solution)
        {
            while (solution.IsSolutionFound() == false)
            {
                // The MHSSolver has ended in an unsatisfiable assignment
                if (solution.Assignments.Count == solution.VariableLookup.Count)
                    return false;

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

            return true;
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
            int max_unsat_clauses = 0;

            foreach (string variable in variables)
            {
                if (!solution.Assignments.ContainsKey(variable))
                {
                    int unsat_clauses = clauses.Count(x => x.Literals.Select(y => y.Variable).Contains(variable));
                    if (unsat_clauses > max_unsat_clauses)
                    {
                        max_unsat_clauses = unsat_clauses;
                        chosen = variable;
                    }
                }
            }

            solution.SetAssignment(chosen, clauses.First().Literals.First().Value);
        }
    }
}