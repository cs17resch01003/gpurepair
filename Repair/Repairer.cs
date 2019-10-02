using GPURepair.Repair.Diagnostics;
using GPURepair.Repair.Exceptions;
using GPURepair.Solvers.Exceptions;
using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;

namespace GPURepair.Repair
{
    public class Repairer
    {
        /// <summary>
        /// The constraint generator.
        /// </summary>
        private ConstraintGenerator constraintGenerator;

        /// <summary>
        /// The barriers.
        /// </summary>
        private Dictionary<string, Barrier> barriers;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public Repairer(string filePath)
        {
            Microsoft.Boogie.Program program = ConstraintGenerator.ReadFile(filePath);
            PopulateBarriers(program);

            constraintGenerator = new ConstraintGenerator(filePath, barriers);
        }

        /// <summary>
        /// Populates the barriers.
        /// </summary>
        /// <param name="program">The given program.</param>
        private void PopulateBarriers(Microsoft.Boogie.Program program)
        {
            barriers = new Dictionary<string, Barrier>();
            Regex regex = new Regex(@"^b\d+$");

            foreach (Declaration declaration in program.TopLevelDeclarations)
                if (declaration is GlobalVariable)
                {
                    GlobalVariable globalVariable = declaration as GlobalVariable;
                    if (regex.IsMatch(globalVariable.Name))
                    {
                        barriers.Add(globalVariable.Name, new Barrier()
                        {
                            Name = globalVariable.Name,
                            Variable = globalVariable
                        });
                    }
                }

            foreach (Implementation implementation in program.Implementations)
                foreach (Block block in implementation.Blocks)
                    foreach (Cmd command in block.Cmds)
                        if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;
                            if (call.callee.Contains("$bugle_barrier"))
                            {
                                int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                                string barrierName = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                                bool generated = ContainsAttribute(call, "repair_instrumented");

                                Barrier barrier = barriers[barrierName];

                                barrier.Implementation = implementation;
                                barrier.Block = block;
                                barrier.Location = location;
                                barrier.Generated = generated;
                            }
                        }

            Logger.Barriers = barriers.Count;
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(out Dictionary<string, bool> assignments)
        {
            List<Error> errors = new List<Error>();
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

                    IEnumerable<Error> current_errors = VerifyProgram(assignments, errors);
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
                catch (SatisfiabilityError)
                {
                    throw new RepairError("The program could not be repaired because of unsatisfiable clauses!");
                }
                catch (AssertionError)
                {
                    assignments = new Dictionary<string, bool>();
                    foreach (Barrier barrier in barriers.Values)
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
        private IEnumerable<Error> VerifyProgram(Dictionary<string, bool> assignments, IEnumerable<Error> errors)
        {
            Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments, errors);

            IEnumerable<Error> current_errors;
            using (Watch watch = new Watch(Measure.Verification))
            {
                Verifier verifier = new Verifier(program, assignments, barriers);
                current_errors = verifier.GetErrors();
            }

            return current_errors;
        }

        /// <summary>
        /// Checks if an call command has the given attribute.
        /// </summary>
        /// <param name="call">The call command.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private bool ContainsAttribute(CallCmd assert, string attribute)
        {
            QKeyValue keyValue = assert.Attributes;
            bool found = false;

            while (keyValue != null && !found)
            {
                if (attribute == keyValue.Key)
                    found = true;
                keyValue = keyValue.Next;
            }

            return found;
        }
    }
}