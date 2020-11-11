namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Common.Diagnostics;
    using GPURepair.Repair.Diagnostics;
    using GPURepair.Repair.Errors;
    using GPURepair.Repair.Exceptions;
    using GPURepair.Repair.Metadata;

    public class Repairer
    {
        /// <summary>
        /// The constraint generator.
        /// </summary>
        private ConstraintGenerator constraintGenerator;

        /// <summary>
        /// Disables inspection of programmer inserted barriers.
        /// </summary>
        private bool disableInspection;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="disableInspection">Disables inspection of programmer inserted barriers.</param>
        /// <param name="useAxioms">Use axioms for instrumentation instead of variable assignments.</param>
        public Repairer(string filePath, bool disableInspection, bool useAxioms)
        {
            this.disableInspection = disableInspection;

            ProgramMetadata.PopulateMetadata(filePath, useAxioms);
            constraintGenerator = new ConstraintGenerator(filePath, useAxioms);
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <param name="defaultType">The default solver to use.</param>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(Solver.SolverType defaultType, out Dictionary<string, bool> assignments)
        {
            List<RepairableError> errors = new List<RepairableError>();
            Dictionary<string, bool> solution = null;

            while (true)
            {
                try
                {
                    Solver.SolverType type = defaultType;
                    Solver solver = new Solver();

                    if (solution == null)
                    {
                        if (!errors.Any() && disableInspection)
                        {
                            // for the first run, disable all the barriers
                            assignments = new Dictionary<string, bool>();
                            foreach (string barrierName in ProgramMetadata.Barriers.Keys)
                                assignments.Add(barrierName, false);
                        }
                        else
                        {
                            // try finding a solution for the errors encountered so far
                            assignments = solver.Solve(errors, ref type);
                        }
                    }
                    else if (solution.Any(x => x.Value == true))
                    {
                        // if the tentative solution has any barriers enabled, optimize it
                        assignments = solver.Optimize(errors, solution, out type);
                    }
                    else
                    {
                        // if the tentative solution has no barriers enabled, it is the optimum solution
                        type = Solver.SolverType.Optimizer;
                        assignments = solution;
                    }

                    // skip the verification if the solution wasn't optimized
                    if (type == Solver.SolverType.Optimizer &&
                        assignments.Count(x => x.Value == true) == solution.Count(x => x.Value == true))
                    {
                        return constraintGenerator.ConstraintProgram(assignments);
                    }

                    IEnumerable<RepairableError> current_errors = VerifyProgram(assignments);
                    if (type == Solver.SolverType.Optimizer)
                    {
                        Logger.Log($"RunsAfterOpt");
                        if (current_errors.Any())
                            Logger.Log($"FailsAfterOpt");
                    }

                    if (!current_errors.Any())
                    {
                        // unassigned barriers are marked as not needed
                        foreach (string barrierName in ProgramMetadata.Barriers.Keys)
                            if (!assignments.ContainsKey(barrierName))
                                assignments.Add(barrierName, false);

                        if (type == Solver.SolverType.MaxSAT || type == Solver.SolverType.Optimizer)
                            return constraintGenerator.ConstraintProgram(assignments);
                        else
                            solution = assignments;
                    }
                    else
                    {
                        solution = null;
                        errors.AddRange(current_errors);
                    }
                }
                catch (AssertionException)
                {
                    assignments = new Dictionary<string, bool>();
                    foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
                        assignments.Add(barrier.Name, !barrier.Generated);

                    IEnumerable<Error> current_errors = VerifyProgram(assignments);
                    if (!current_errors.Any())
                        return constraintGenerator.ConstraintProgram(assignments);

                    throw;
                }
            }
        }

        /// <summary>
        /// Verifies the program and returns the errors.
        /// </summary>
        /// <param name="assignments">The assignements</param>
        /// <returns>The errors.</returns>
        private IEnumerable<RepairableError> VerifyProgram(Dictionary<string, bool> assignments)
        {
            Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments);

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
