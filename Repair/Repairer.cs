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
        /// <param name="assignments">The assignments to the variables.</param>
        /// <returns>The fixed program if one exists.</returns>
        public Microsoft.Boogie.Program Repair(out Dictionary<string, bool> assignments)
        {
            List<Error> errors = new List<Error>();
            bool repaired = false;

            while (true)
            {
                Solver solver = new Solver();
                assignments = repaired ? solver.OptimizedSolve(errors) : solver.Solve(errors);

                Microsoft.Boogie.Program program = ReadFile(filePath);

                ConstraintGenerator constraintGenerator = new ConstraintGenerator(program);
                constraintGenerator.ConstraintProgram(errors, assignments);

                if (repaired)
                    return program;

                string tempFile = WriteFile(program);
                program = ReadFile(tempFile);

                Error error;
                using (Watch watch = new Watch())
                {
                    Verifier verifier = new Verifier(program);
                    error = verifier.GetError();
                }

                File.Delete(tempFile);
                if (error == null)
                    repaired = true;
                else
                    errors.Add(error);
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
    }
}