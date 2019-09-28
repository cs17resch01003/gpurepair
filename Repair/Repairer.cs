using GPURepair.Repair.Exceptions;
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
            constraintGenerator = new ConstraintGenerator(filePath);

            Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(null);
            PopulateBarriers(program);
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

            IEnumerable<Cmd> commands = program.Implementations.SelectMany(x => x.Blocks).SelectMany(x => x.Cmds);
            foreach (Cmd command in commands)
                if (command is CallCmd)
                {
                    CallCmd call = command as CallCmd;
                    if (call.callee.Contains("$bugle_barrier"))
                    {
                        int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                        string barrier = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                        bool generated = ContainsAttribute(call, "repair_instrumented");

                        barriers[barrier].Location = location;
                        barriers[barrier].Generated = generated;
                    }
                }
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(out Dictionary<string, bool> assignments)
        {
            List<Error> errors = new List<Error>();
            while (true)
            {
                try
                {
                    Solver solver = new Solver();
                    assignments = solver.OptimizedSolve(errors);

                    IEnumerable<Error> current_errors = VerifyProgram(assignments);
                    if (!current_errors.Any())
                        return constraintGenerator.ConstraintProgram(assignments);
                    else
                        errors.AddRange(current_errors);
                }
                catch (AssertionError)
                {
                    assignments = new Dictionary<string, bool>();
                    foreach (Barrier barrier in barriers.Values)
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
        private IEnumerable<Error> VerifyProgram(Dictionary<string, bool> assignments)
        {
            Microsoft.Boogie.Program program = constraintGenerator.ConstraintProgram(assignments);

            IEnumerable<Error> current_errors;
            using (Watch watch = new Watch())
            {
                Verifier verifier = new Verifier(program, assignments);
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