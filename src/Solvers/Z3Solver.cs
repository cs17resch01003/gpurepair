namespace GPURepair.Solvers
{
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.Z3;

    public abstract class Z3Solver
    {
        /// <summary>
        /// Generates variables for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The variables.</returns>
        protected Dictionary<string, Z3Variable> GetVariables(Context context, IEnumerable<Clause> input)
        {
            Dictionary<string, Z3Variable> variables = new Dictionary<string, Z3Variable>();

            IEnumerable<string> variableNames = input.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct();
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
        protected List<BoolExpr> GenerateClauses(Context context, IEnumerable<Clause> input,
            Dictionary<string, Z3Variable> variables)
        {
            List<BoolExpr> clauses = new List<BoolExpr>();
            foreach (Clause clause in input)
                clauses.Add(GenerateClause(context, clause, variables));

            return clauses;
        }

        /// <summary>
        /// Generates clauses for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="input">The clauses.</param>
        /// <param name="variables">The variables.</param>
        /// <returns>The Z3 clauses.</returns>
        protected Dictionary<BoolExpr, uint> GenerateClauses(Context context, Dictionary<Clause, uint> input,
            Dictionary<string, Z3Variable> variables)
        {
            Dictionary<BoolExpr, uint> clauses = new Dictionary<BoolExpr, uint>();
            foreach (Clause clause in input.Keys)
                clauses.Add(GenerateClause(context, clause, variables), input[clause]);

            return clauses;
        }

        /// <summary>
        /// Generates a clause for the Z3 solver.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="clause">The input clause.</param>
        /// <param name="variables">The variables.</param>
        /// <returns>The Z3 clause.</returns>
        private BoolExpr GenerateClause(Context context, Clause clause, Dictionary<string, Z3Variable> variables)
        {
            List<BoolExpr> literals = new List<BoolExpr>();
            foreach (Literal literal in clause.Literals)
            {
                Z3Variable variable = variables[literal.Variable];
                literals.Add(literal.Value ? variable.Positive : variable.Negative);
            }

            return context.MkOr(literals);
        }

        /// <summary>
        /// Represents a variable with its positive and negative Z3 representation.
        /// </summary>
        protected class Z3Variable
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
