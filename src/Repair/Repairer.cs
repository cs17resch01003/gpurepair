using System.Collections.Generic;
using System.Linq;
using GPURepair.Repair.Diagnostics;
using GPURepair.Repair.Errors;
using GPURepair.Repair.Exceptions;
using GPURepair.Repair.Metadata;

namespace GPURepair.Repair
{
    public class Repairer
    {
        /// <summary>
        /// The constraint generator.
        /// </summary>
        private ConstraintGenerator constraintGenerator;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public Repairer(string filePath)
        {
            ProgramMetadata.PopulateMetadata(filePath);
            constraintGenerator = new ConstraintGenerator(filePath);
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(out Dictionary<string, bool> assignments)
        {
            List<RepairableError> errors = new List<RepairableError>();
            Dictionary<string, bool> solution = null;

            while (true)
            {
                try
                {
                    Solver.SolverType type;

                    Solver solver = new Solver();
                    assignments = solution == null ? solver.Solve(errors, out type) : solver.Optimize(errors, solution, out type);

                    // skip the verification if the solution wasn't optimized
                    if (type == Solver.SolverType.Optimizer && assignments.Count(x => x.Value == true) == solution.Count(x => x.Value == true))
                        return constraintGenerator.ConstraintProgram(assignments, errors);

                    IEnumerable<RepairableError> current_errors = VerifyProgram(assignments, errors);
                    if (type == Solver.SolverType.Optimizer)
                    {
                        Logger.VerifierRunsAfterOptimization++;
                        if (current_errors.Any())
                            Logger.VerifierFailuresAfterOptimization++;
                    }

                    if (!current_errors.Any())
                    {
                        if (type == Solver.SolverType.MaxSAT || type == Solver.SolverType.Optimizer)
                            return constraintGenerator.ConstraintProgram(assignments, errors);
                        else
                            solution = assignments;
                    }
                    else
                    {
                        solution = null;
                        errors.AddRange(current_errors);
                    }
                }
                catch (AssertionError)
                {
                    assignments = new Dictionary<string, bool>();
                    foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                        assignments.Add(barrier.Name, !barrier.Generated);

                    IEnumerable<Error> current_errors = VerifyProgram(assignments, errors);
                    if (!current_errors.Any())
                        return constraintGenerator.ConstraintProgram(assignments, errors);

                    throw;
                }
            }
        }

        /// <summary>
        /// Verifies the program and returns the errors.
        /// </summary>
        /// <param name="assignments">The assignements</param>
        /// <param name="errors">The errors.</param>
        /// <returns>The errors.</returns>
        private IEnumerable<RepairableError> VerifyProgram(
            Dictionary<string, bool> assignments, IEnumerable<RepairableError> errors)
        {
            Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments, errors);

            IEnumerable<RepairableError> current_errors;
            using (Watch watch = new Watch(Measure.Verification))
            {
                Verifier verifier = new Verifier(program, assignments);
                current_errors = verifier.GetErrors();
            }

            return current_errors;
        }
    }
}
