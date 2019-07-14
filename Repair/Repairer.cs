using GPURepair.Solvers;
using Microsoft.Boogie;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GPURepair.Repair
{
    public class Repairer
    {
        /// <summary>
        /// The file path.
        /// </summary>
        private string filePath;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public Repairer(string filePath)
        {
            this.filePath = filePath;

            Microsoft.Boogie.Program program = ReadFile(filePath);
            LogBarriers(program);
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <param name="assignments">The assignments to the variables.</param>
        /// <param name="efficientSolving">Specifies if the MHS solver should be used or the regular SAT Solver.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(out Dictionary<string, bool> assignments, bool efficientSolving = true)
        {
            List<Error> errors = new List<Error>();
            bool repaired = false;

            while (true)
            {
                Solver solver = new Solver();
                assignments = repaired ? solver.OptimizedSolve(errors) : solver.Solve(errors, efficientSolving);

                Microsoft.Boogie.Program program = ReadFile(filePath);

                ConstraintGenerator constraintGenerator = new ConstraintGenerator(program);
                constraintGenerator.ConstraintProgram(assignments);

                if (repaired)
                    return program;

                string tempFile = WriteFile(program);
                program = ReadFile(tempFile);

                IEnumerable<Error> current_errors;
                using (Watch watch = new Watch())
                {
                    Verifier verifier = new Verifier(program, assignments);
                    current_errors = verifier.GetErrors();
                }

                File.Delete(tempFile);
                if (!current_errors.Any())
                    repaired = true;
                else
                    errors.AddRange(current_errors);
            }
        }

        /// <summary>
        /// Reads the file and generates a program.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns>The program.</returns>
        private Microsoft.Boogie.Program ReadFile(string filePath)
        {
            Microsoft.Boogie.Program program;
            Parser.Parse(filePath, new List<string>() { "FILE_0" }, out program);

            ResolutionContext context = new ResolutionContext(null);
            program.Resolve(context);

            program.Typecheck();
            new ModSetCollector().DoModSetAnalysis(program);

            return program;
        }

        /// <summary>
        /// Writes the program to a temp file.
        /// </summary>
        /// <param name="program">The program.</param>
        /// <returns>The temp file path.</returns>
        private string WriteFile(Microsoft.Boogie.Program program)
        {
            string tempFile = filePath.Replace(".cbpl", ".temp.cbpl");
            using (TokenTextWriter writer = new TokenTextWriter(tempFile, true))
                program.Emit(writer);

            return tempFile;
        }

        /// <summary>
        /// Logs the number of barriers.
        /// </summary>
        /// <param name="program">The program.</param>
        private void LogBarriers(Microsoft.Boogie.Program program)
        {
            List<string> barriers = new List<string>();

            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    foreach (Cmd command in block.Cmds)
                    {
                        if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;
                            if (call.callee.Contains("$bugle_barrier"))
                            {
                                int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                                string barrier = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                                bool generated = ContainsAttribute(call, "repair_instrumented");

                                if (location != -1 && generated == true)
                                    barriers.Add(barrier);
                            }
                        }
                    }
                }
            }

            Logger.Barriers = barriers.Distinct().Count();
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