using Microsoft.Z3;
using System;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Repair
{
    public class Solver
    {
        /// <summary>
        /// Determine the barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<Error> errors)
        {
            // Use Z3 and figure out the variable assignments
            using (Context context = new Context())
            {
                Dictionary<string, dynamic> variables = GetVariables(errors, context);
                List<BoolExpr> clauses = GenerateClauses(errors, variables, context);

                if (SAT(clauses, context) == false)
                    throw new Exception("The program cannot be repaired!");

                UnitPropogation(clauses);
            }

            Dictionary<string, bool> assignments = new Dictionary<string, bool>();
            if (errors.Any())
                assignments.Add("b1", true);

            return assignments;
        }

        /// <summary>
        /// Determine the optimized barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> OptimizedSolve(List<Error> errors)
        {
            Dictionary<string, bool> assignments = new Dictionary<string, bool>();
            if (errors.Any())
                assignments.Add("b1", true);

            return assignments;
        }

        /// <summary>
        /// Generates variables for the Z3 solver.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The variables.</returns>
        private Dictionary<string, dynamic> GetVariables(List<Error> errors, Context context)
        {
            Dictionary<string, dynamic> variables = new Dictionary<string, dynamic>();

            IEnumerable<string> variableNames = errors.SelectMany(x => x.Variables).Select(x => x.Name).Distinct();
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
        /// <param name="errors">The errors.</param>
        /// <param name="variables">The variables.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>The clauses.</returns>
        private List<BoolExpr> GenerateClauses(List<Error> errors, Dictionary<string, dynamic> variables, Context context)
        {
            List<BoolExpr> clauses = new List<BoolExpr>();
            foreach (Error error in errors)
            {
                IEnumerable<BoolExpr> results = from errorVariable in error.Variables
                                                join variable in variables
                                                on errorVariable.Name equals variable.Key
                                                select (BoolExpr)(error.ErrorType == ErrorType.Race ? variable.Value.Positive : variable.Value.Negative);

                BoolExpr clause = context.MkOr(results);
                clauses.Add(clause);
            }

            return clauses;
        }

        /// <summary>
        /// Determines if the given clauses can be satisfiable.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        /// <param name="context">The Z3 context.</param>
        /// <returns>true if the clauses can be satisfied and false otherwise.</returns>
        private bool SAT(List<BoolExpr> clauses, Context context)
        {
            Microsoft.Z3.Solver solver = context.MkSolver();
            Status status = solver.Check(clauses);

            if (status == Status.SATISFIABLE)
                return true;
            return false;
        }

        private void UnitPropogation(List<BoolExpr> clauses)
        {
            foreach (BoolExpr clause in clauses)
            {
            }
        }
    }
}