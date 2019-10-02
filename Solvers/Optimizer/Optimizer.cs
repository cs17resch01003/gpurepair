using GPURepair.Solvers.Exceptions;
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
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve(int limit)
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

                Status status = solver.Check();
                if (status == Status.SATISFIABLE)
                {
                    Dictionary<string, bool> assignments = new Dictionary<string, bool>();
                    foreach (string variable in variables.Keys)
                    {
                        Expr expr = solver.Model.Evaluate(variables[variable].Positive);
                        if (expr.BoolValue != Z3_lbool.Z3_L_UNDEF)
                            assignments.Add(variable, expr.BoolValue == Z3_lbool.Z3_L_TRUE ? true : false);
                    }

                    return assignments;
                }

                throw new SatisfiabilityError();
            }
        }
    }
}
