using Microsoft.Z3;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class SATSolver
    {
        /// <summary>
        /// The clauses.
        /// </summary>
        public List<Clause> Clauses { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        public SATSolver(List<Clause> clauses)
        {
            Clauses = clauses;
        }

        /// <summary>
        /// Solves the given clauses.
        /// </summary>
        /// <returns>The assignments if the clauses can be satisfied and null otherwise.</returns>
        public Dictionary<string, bool> Solve()
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, dynamic> variables = GetVariables(context);
                List<BoolExpr> clauses = GenerateClauses(variables, context);

                return SAT(variables, clauses, context);
            }
        }

        /// <summary>
        /// Generates variables for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The variables.</returns>
        private Dictionary<string, dynamic> GetVariables(Context context)
        {
            Dictionary<string, dynamic> variables = new Dictionary<string, dynamic>();

            IEnumerable<string> variableNames = Clauses.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct();
            foreach (string variableName in variableNames)
            {
                BoolExpr positive = context.MkBoolConst(variableName);
                BoolExpr negative = context.MkNot(positive);

                variables.Add(variableName, new
                {
                    VariableName = variableName,
                    Positive = positive,
                    Negative = negative
                });
            }

            return variables;
        }

        /// <summary>
        /// Generates clauses for the Z3 solver.
        /// </summary>
        /// <param name="variables">The variables.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The Z3 clauses.</returns>
        private List<BoolExpr> GenerateClauses(Dictionary<string, dynamic> variables, Context context)
        {
            List<BoolExpr> clauses = new List<BoolExpr>();
            foreach (Clause clause in Clauses)
            {
                List<BoolExpr> literals = new List<BoolExpr>();
                foreach (Literal literal in clause.Literals)
                {
                    dynamic variable = variables[literal.Variable];
                    literals.Add(literal.Value ? variable.Positive : variable.Negative);
                }

                clauses.Add(context.MkOr(literals));
            }

            return clauses;
        }

        /// <summary>
        /// Determines if the given clauses can be satisfiable.
        /// </summary>
        /// <param name="variables">The variables.</param>
        /// <param name="clauses">The clauses.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The assignments if the clauses can be satisfied and null otherwise.</returns>
        private Dictionary<string, bool> SAT(Dictionary<string, dynamic> variables, List<BoolExpr> clauses, Context context)
        {
            Dictionary<string, bool> assignments = null;

            Solver solver = context.MkSolver();
            solver.Assert(clauses.ToArray());

            Status status = solver.Check();
            if (status == Status.SATISFIABLE)
            {
                assignments = new Dictionary<string, bool>();
                foreach (string variable in variables.Keys)
                {
                    Expr expr = solver.Model.Evaluate(variables[variable].Positive);
                    if (expr.BoolValue != Z3_lbool.Z3_L_UNDEF)
                        assignments.Add(variable, expr.BoolValue == Z3_lbool.Z3_L_TRUE ? true : false);
                }
            }

            return assignments;
        }
    }
}