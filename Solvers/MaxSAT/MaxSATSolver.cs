using Microsoft.Z3;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class MaxSATSolver
    {
        /// <summary>
        /// The hard clauses.
        /// </summary>
        public List<Clause> HardClauses { get; set; }

        /// <summary>
        /// The soft clauses.
        /// </summary>
        public List<Clause> SoftClauses { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="hard_clauses">The hard clauses.</param>
        /// <param name="soft_clauses">The soft clauses.</param>
        public MaxSATSolver(List<Clause> hard_clauses, List<Clause> soft_clauses)
        {
            HardClauses = hard_clauses;
            SoftClauses = soft_clauses;
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve()
        {
            int relaxed = 0;
            int limit = SoftClauses.Count;

            while (relaxed <= limit)
            {
                Dictionary<string, bool> solution = Solve(relaxed);
                if (solution != null)
                    return solution;

                relaxed += 1;
            }

            throw new Exception("The given clauses cannot be satisfied!");
        }

        /// <summary>
        /// Solves the clauses and returns a solution.
        /// </summary>
        /// <returns>The solution.</returns>
        public Dictionary<string, bool> Solve(int relaxed)
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, Z3Variable> variables = GetVariables(context);

                List<BoolExpr> hard_clauses = GenerateClauses(context, HardClauses, variables);
                List<BoolExpr> soft_clauses = GenerateClauses(context, SoftClauses, variables);

                List<BoolExpr> final_clauses;
                if (relaxed == 0)
                {
                    final_clauses = hard_clauses.Union(soft_clauses).ToList();
                }
                else
                {
                    Dictionary<string, Z3Variable> relaxation_variables = GetRelaxationVariables(context, soft_clauses.Count);
                    soft_clauses = AddRelaxationVariables(context, soft_clauses, relaxation_variables);

                    List<BoolExpr> relaxation_clauses = GenerateConstraints(context, relaxation_variables, relaxed);
                    final_clauses = hard_clauses.Union(soft_clauses).Union(relaxation_clauses).ToList();
                }

                Model model = SAT(final_clauses, context);
                if (model == null)
                    return null;

                Dictionary<string, bool> assignments = new Dictionary<string, bool>();
                foreach (string variable in variables.Keys)
                {
                    BoolExpr key = variables[variable].Positive;
                    if (model.Evaluate(key).BoolValue == Z3_lbool.Z3_L_TRUE)
                        assignments.Add(variable, true);
                    else
                        assignments.Add(variable, false);
                }

                return assignments;
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

            IEnumerable<string> variableNames = HardClauses.Union(SoftClauses)
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
        /// Generates relaxation variables for MaxSAT.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="count">The number of soft clauses.</param>
        /// <returns>The relaxation variables.</returns>
        private Dictionary<string, Z3Variable> GetRelaxationVariables(Context context, int count)
        {
            Dictionary<string, Z3Variable> relaxation_variables = new Dictionary<string, Z3Variable>();
            for (int i = 0; i < count; i++)
            {
                string variableName = string.Format("r_{0}", i);
                relaxation_variables.Add(variableName, new Z3Variable(context, variableName));
            }

            return relaxation_variables;
        }

        /// <summary>
        /// Adds the relaxation variable to the soft clauses.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="soft_clauses">The soft clauses.</param>
        /// <param name="relaxation_variables">The relaxation variables.</param>
        /// <returns>The Z3 soft clauses.</returns>
        private List<BoolExpr> AddRelaxationVariables(Context context, List<BoolExpr> soft_clauses,
             Dictionary<string, Z3Variable> relaxation_variables)
        {
            List<BoolExpr> relaxed_clauses = new List<BoolExpr>();

            for (int i = 0; i < soft_clauses.Count; i++)
            {
                BoolExpr clause = soft_clauses[i];
                Z3Variable variable = relaxation_variables[string.Format("r_{0}", i)];

                relaxed_clauses.Add(context.MkOr(clause, variable.Positive));
            }

            return relaxed_clauses;
        }

        /// <summary>
        /// Generates constraints to restrict the relaxation variables.
        /// Implements the Sequential Counter Encoding scheme - http://www.cs.toronto.edu/~fbacchus/csc2512/at_most_k.pdf.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="relaxation_variables">The relaxation variables.</param>
        /// <param name="relaxed">The relaxation limit.</param>
        /// <returns>The constraints to restrict the relaxation variables.</returns>
        private List<BoolExpr> GenerateConstraints(Context context,
            Dictionary<string, Z3Variable> relaxation_variables, int relaxed)
        {
            List<BoolExpr> inputs = relaxation_variables.Select(x => x.Value.Negative).ToList();
            Z3Variable[,] registers = CreateRegisterVariables(context, inputs.Count, relaxed);

            int n = inputs.Count;
            int k = relaxed;

            List<BoolExpr> clauses = new List<BoolExpr>();

            // ensures that if the variable is true,
            // the first bit of the register is true
            for (int i = 0; i < n - 1; i++)
                clauses.Add(context.MkOr(inputs[i], registers[i, 0].Positive));

            // only the first bit of the first register can be true
            for (int j = 1; j < k; j++)
                clauses.Add(context.MkOr(registers[0, j].Negative));

            // contain the value of the previous register
            // plus the current variable
            for (int i = 1; i < n - 1; i++)
                for (int j = 0; j < k; j++)
                    clauses.Add(context.MkOr(registers[i - 1, j].Negative, registers[i, j].Positive));

            // contain the value of the previous register
            // plus the current variable
            for (int i = 1; i < n - 1; i++)
                for (int j = 1; j < k; j++)
                    clauses.Add(context.MkOr(inputs[i], registers[i - 1, j - 1].Negative, registers[i, j].Positive));

            // asserts that there is no overflow
            for (int i = 1; i < n; i++)
                clauses.Add(context.MkOr(inputs[i], registers[i - 1, relaxed - 1].Negative));

            return clauses;
        }

        /// <summary>
        /// Creates the register variables based on the number of variables and the relaxation limit.
        /// </summary>
        /// <param name="context">The Z3 context.</param>
        /// <param name="variables">The number of variables.</param>
        /// <param name="relaxed">The relaxation limit.</param>
        /// <returns>The register variables.</returns>
        private Z3Variable[,] CreateRegisterVariables(Context context, int variables, int relaxed)
        {
            Z3Variable[,] registers = new Z3Variable[variables, relaxed];
            for (int i = 0; i < variables; i++)
                for (int j = 0; j < relaxed; j++)
                {
                    string variableName = string.Format("reg_{0}_{1}", i, j);
                    registers[i, j] = new Z3Variable(context, variableName);
                }

            return registers;
        }

        /// <summary>
        /// Determines if the given clauses can be satisfiable.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The SAT model.</returns>
        private Model SAT(List<BoolExpr> clauses, Context context)
        {
            Solver solver = context.MkSolver();
            Status status = solver.Check(clauses);

            if (status != Status.SATISFIABLE)
                return null;

            return solver.Model;
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