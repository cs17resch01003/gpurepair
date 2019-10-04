using Microsoft.Z3;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers.Optimizer
{
    public class Optimizer : Z3Solver
    {
        /// <summary>
        /// The hard clauses.
        /// </summary>
        private List<Clause> clauses { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        public Optimizer(List<Clause> clauses)
        {
            this.clauses = clauses;
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <param name="status">The solver status.</param>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve(int limit, out SolverStatus status)
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, Z3Variable> variables = GetVariables(context, this.clauses);
                List<BoolExpr> clauses = GenerateClauses(context, this.clauses, variables);

                Solver solver = context.MkSolver();
                solver.Assert(clauses.ToArray());

                BoolExpr condition = context.MkPBLe(
                    variables.Values.Select(x => 1).ToArray(),
                    variables.Values.Select(x => x.Positive).ToArray(),
                    limit);
                solver.Assert(condition);

                Status solver_status = solver.Check();
                if (solver_status == Status.SATISFIABLE)
                {
                    Dictionary<string, bool> assignments = new Dictionary<string, bool>();
                    foreach (string variable in variables.Keys)
                    {
                        Expr expr = solver.Model.Evaluate(variables[variable].Positive);
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
