using GPURepair.Solvers;
using GPURepair.Solvers.Exceptions;
using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Repair
{
    public class Solver
    {
        /// <summary>
        /// Determine the optimized barrier assignments based on the error traces.
        /// </summary>
        /// <param name="errors">The errors.</param>
        /// <returns>The barrier assignments.</returns>
        public Dictionary<string, bool> Solve(List<Error> errors)
        {
            using (Watch watch = new Watch(Watch.Measure.MaxSAT_SolutionTime))
            {
                // fall back to MaxSAT if MHS fails
                List<Clause> clauses = GenerateClauses(errors);
                List<string> variables = clauses.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct().ToList();

                List<Clause> soft_clauses = GenerateSoftClauses(variables);
                MaxSATSolver solver = new MaxSATSolver(clauses, soft_clauses);

                return solver.Solve();
            }

            //try
            //{
            //    using (Watch watch = new Watch(Watch.Measure.MHS_SolutionTime))
            //    {
            //        List<Clause> clauses = GenerateClauses(errors);
            //        MHSSolver solver = new MHSSolver(clauses);

            //        return solver.Solve();
            //    }
            //}
            //catch (SatisfiabilityError)
            //{
            //    using (Watch watch = new Watch(Watch.Measure.MaxSAT_SolutionTime))
            //    {
            //        // fall back to MaxSAT if MHS fails
            //        List<Clause> clauses = GenerateClauses(errors);
            //        List<string> variables = clauses.SelectMany(x => x.Literals).Select(x => x.Variable).Distinct().ToList();

            //        List<Clause> soft_clauses = GenerateSoftClauses(variables);
            //        MaxSATSolver solver = new MaxSATSolver(clauses, soft_clauses);

            //        return solver.Solve();
            //    }
            //}
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
                bool value = error.ErrorType == ErrorType.Race;

                Clause clause = new Clause();
                foreach (Variable variable in error.Variables)
                    clause.Add(new Literal(variable.Name, value));

                clauses.Add(clause);
            }

            return clauses;
        }

        /// <summary>
        /// Generates soft clauses based on the variables.
        /// </summary>
        /// <param name="variables">The variables.</param>
        /// <returns>The soft clauses.</returns>
        private List<Clause> GenerateSoftClauses(List<string> variables)
        {
            List<Clause> clauses = new List<Clause>();
            foreach (string variable in variables)
            {
                Clause clause = new Clause();
                clause.Add(new Literal(variable, false));

                clauses.Add(clause);
            }

            return clauses;
        }
    }
}