namespace GPURepair.Repair
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Common.Diagnostics;
    using GPURepair.Repair.Diagnostics;
    using GPURepair.Repair.Errors;
    using GPURepair.Repair.Exceptions;
    using GPURepair.Repair.Metadata;
    using GPURepair.Repair.Verifiers;

    public class Repairer : IDisposable
    {
        /// <summary>
        /// The constraint generator.
        /// </summary>
        private ConstraintGenerator constraintGenerator;

        /// <summary>
        /// The incremental verifier.
        /// </summary>
        private IncrementalVerifier incrementalVerifier;

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
        /// <param name="verificationType">The verification type.</param>
        /// <param name="solverType">The default solver to use.</param>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(
            VerificationType verificationType,
            Solver.SolverType solverType,
            out Dictionary<string, bool> assignments)
        {
            List<RepairableError> errors = new List<RepairableError>();
            Dictionary<string, bool> solution = null;

            if (disableInspection)
            {
                // for the first run, disable all the barriers
                // this will mimic what AutoSync does
                assignments = new Dictionary<string, bool>();
                foreach (string barrierName in ProgramMetadata.Barriers.Keys)
                    assignments.Add(barrierName, false);

                IEnumerable<RepairableError> current_errors = VerifyProgram(VerificationType.Classic, assignments);
                if (!current_errors.Any())
                    return constraintGenerator.ConstraintProgram(assignments);

                // the errors identified in this run can be reused only for classic verification
                // incremental verification needs the barrier variables to be free the first time
                if (verificationType == VerificationType.Classic)
                    errors.AddRange(current_errors);
            }

            while (true)
            {
                try
                {
                    Solver.SolverType type = solverType;
                    Solver solver = new Solver();

                    if (solution == null)
                    {
                        // try finding a solution for the errors encountered so far
                        assignments = solver.Solve(errors, ref type);
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

                    IEnumerable<RepairableError> current_errors = VerifyProgram(verificationType, assignments);
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

                    IEnumerable<RepairableError> current_errors = VerifyProgram(verificationType, assignments);
                    if (!current_errors.Any())
                        return constraintGenerator.ConstraintProgram(assignments);

                    throw;
                }
            }
        }

        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            if (incrementalVerifier != null)
                incrementalVerifier.Dispose();
        }

        /// <summary>
        /// Verifies the program and returns the errors.
        /// </summary>
        /// <param name="verificationType">The verification type.</param>
        /// <param name="assignments">The assignements</param>
        /// <returns>The errors.</returns>
        private IEnumerable<RepairableError> VerifyProgram(
            VerificationType verificationType,
            Dictionary<string, bool> assignments)
        {
            IEnumerable<RepairableError> current_errors;
            using (Watch watch = new Watch(Measure.Verification))
            {
                if (verificationType == VerificationType.Classic)
                {
                    Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments);

                    ClassicVerifier verifier = new ClassicVerifier(program);
                    current_errors = verifier.GetErrors(assignments);
                }
                else
                {
                    if (incrementalVerifier == null)
                    {
                        Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments);
                        incrementalVerifier = new IncrementalVerifier(program);
                    }

                    current_errors = incrementalVerifier.GetErrors(assignments);
                }
            }

            return current_errors;
        }

        /// <summary>
        /// The verification type used.
        /// </summary>
        public enum VerificationType
        {
            Classic,
            Incremental
        }
    }
}
