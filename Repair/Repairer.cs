using Microsoft.Boogie;
using System;
using System.Collections.Generic;
using System.IO;

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
        }

        /// <summary>
        /// Reapirs the program at the given filePath.
        /// </summary>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair()
        {
            List<Error> errors = new List<Error>();
            while (true)
            {
                Solver solver = new Solver();
                Dictionary<string, bool> assignments = solver.Solve(errors);

                Microsoft.Boogie.Program program = ReadFile(filePath);

                ConstraintGenerator constraintGenerator = new ConstraintGenerator(program);
                constraintGenerator.ConstraintProgram(errors, assignments);

                string tempFile = WriteFile(program);
                program = ReadFile(tempFile);

                Verifier verifier = new Verifier(program);
                Error error = verifier.GetError();

                if (error == null)
                {
                    program = ReadFile(tempFile);
                    File.Delete(tempFile);
                    return program;
                }
                else
                {
                    File.Delete(tempFile);
                    errors.Add(error);
                }
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
            string tempFile = filePath + "." + DateTime.Now.ToString("hhmmss") + ".temp";
            using (TokenTextWriter writer = new TokenTextWriter(tempFile, true))
                program.Emit(writer);

            return tempFile;
        }
    }
}