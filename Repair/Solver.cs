using GPURepair.Solvers;
using Microsoft.Boogie;
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
            List<Clause> clauses = GenerateClauses(errors);

            SATSolver solver = new SATSolver(clauses);
            bool result = solver.IsSatisfiable();

            if (result == false)
                throw new Exception("The program cannot be repaired!");

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
        /// Generates clauses based on the errors.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The clauses.</returns>
        private List<Clause> GenerateClauses(List<Error> errors)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (Error error in errors)
            {
                bool value = error.ErrorType == ErrorType.Divergence;

                Clause clause = new Clause();
                foreach (Variable variable in error.Variables)
                    clause.Add(new Literal(variable.Name, value));
            }

            return clauses;
        }
    }
}