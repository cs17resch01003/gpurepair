using Microsoft.Z3;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class MaxSATSolver : Z3Solver
    {
        /// <summary>
        /// The hard clauses.
        /// </summary>
        private List<Clause> hard_clauses { get; set; }

        /// <summary>
        /// The soft clauses.
        /// </summary>
        private List<Clause> soft_clauses { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="hard_clauses">The hard clauses.</param>
        /// <param name="soft_clauses">The soft clauses.</param>
        public MaxSATSolver(List<Clause> hard_clauses, List<Clause> soft_clauses)
        {
            this.hard_clauses = hard_clauses;
            this.soft_clauses = soft_clauses;
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve(out SolverStatus status)
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, Z3Variable> variables = GetVariables(context, this.hard_clauses.Union(this.soft_clauses));

                List<BoolExpr> hard_clauses = GenerateClauses(context, this.hard_clauses, variables);
                List<BoolExpr> soft_clauses = GenerateClauses(context, this.soft_clauses, variables);

                Optimize optimize = context.MkOptimize();
                optimize.Assert(hard_clauses);
                foreach (BoolExpr clause in soft_clauses)
                    optimize.AssertSoft(clause, 1, "group");

                Status solver_status = optimize.Check();
                if (solver_status == Status.SATISFIABLE)
                {
                    Dictionary<string, bool> assignments = new Dictionary<string, bool>();
                    foreach (string variable in variables.Keys)
                    {
                        Expr expr = optimize.Model.Evaluate(variables[variable].Positive);
                        if (expr.BoolValue != Z3_lbool.Z3_L_UNDEF)
                            assignments.Add(variable, expr.BoolValue == Z3_lbool.Z3_L_TRUE ? true : false);
                    }

                    status = SolverStatus.Satisfiable;
                    return assignments;
                }

                status = SolverStatus.Unsatisfiable;
                return null;
            }
        }
    }
}
