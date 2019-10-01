using GPURepair.Solvers.Exceptions;
using Microsoft.Z3;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class MaxSATSolver
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
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve()
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, Z3Variable> variables = GetVariables(context);

                List<BoolExpr> hard_clauses = GenerateClauses(context, this.hard_clauses, variables);
                List<BoolExpr> soft_clauses = GenerateClauses(context, this.soft_clauses, variables);

                Optimize optimize = context.MkOptimize();
                optimize.Assert(hard_clauses);
                foreach (BoolExpr clause in soft_clauses)
                    optimize.AssertSoft(clause, 1, "group");

                Status status = optimize.Check();
                if (status == Status.SATISFIABLE)
                {
                    Dictionary<string, bool> assignments = new Dictionary<string, bool>();
                    foreach (string variable in variables.Keys)
                    {
                        Expr expr = optimize.Model.Evaluate(variables[variable].Positive);
                        if (expr.BoolValue != Z3_lbool.Z3_L_UNDEF)
                            assignments.Add(variable, expr.BoolValue == Z3_lbool.Z3_L_TRUE ? true : false);
                    }

                    return assignments;
                }

                throw new SatisfiabilityError();
            }
        }

        /// <summary>
        /// Generates variables for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The variables.</returns>
        private Dictionary<string, Z3Variable> GetVariables(Context context)
        {
            Dictionary<string, Z3Variable> variables = new Dictionary<string, Z3Variable>();

            IEnumerable<string> variableNames = hard_clauses.Union(soft_clauses)
                .SelectMany(x => x.Literals).Select(x => x.Variable).Distinct();
            foreach (string variableName in variableNames)
                variables.Add(variableName, new Z3Variable(context, variableName));

            return variables;
        }

        /// <summary>
        /// Generates clauses for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="input">The clauses.</param>
        /// <param name="variables">The variables.</param>
        /// <returns>The Z3 clauses.</returns>
        private List<BoolExpr> GenerateClauses(Context context, List<Clause> input,
            Dictionary<string, Z3Variable> variables)
        {
            List<BoolExpr> clauses = new List<BoolExpr>();
            foreach (Clause clause in input)
            {
                List<BoolExpr> literals = new List<BoolExpr>();
                foreach (Literal literal in clause.Literals)
                {
                    Z3Variable variable = variables[literal.Variable];
                    literals.Add(literal.Value ? variable.Positive : variable.Negative);
                }

                clauses.Add(context.MkOr(literals));
            }

            return clauses;
        }

        /// <summary>
        /// Represents a variable with its positive and negative Z3 representation.
        /// </summary>
        private class Z3Variable
        {
            /// <summary>
            /// The variable name.
            /// </summary>
            public string VariableName;

            /// <summary>
            /// The positive representation.
            /// </summary>
            public BoolExpr Positive;

            /// <summary>
            /// The negative representation.
            /// </summary>
            public BoolExpr Negative;

            /// <summary>
            /// The constructor.
            /// </summary>
            /// <param name="context">The Z3 context.</param>
            /// <param name="variableName">The variable name.</param>
            public Z3Variable(Context context, string variableName)
            {
                VariableName = variableName;
                Positive = context.MkBoolConst(variableName);
                Negative = context.MkNot(Positive);
            }
        }
    }
}