namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.IO;
    using Microsoft.Boogie;

    public static class BoogieUtilities
    {
        /// <summary>
        /// Reads the file and generates a program.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns>The program.</returns>
        public static Microsoft.Boogie.Program ReadFile(string filePath)
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
        /// <param name="filePath">The file path.</param>
        /// <param name="program">The program.</param>
        /// <returns>The temp file path.</returns>
        public static string WriteFile(string filePath, Microsoft.Boogie.Program program)
        {
            string tempFile = filePath.Replace(".cbpl", ".temp.cbpl");
            if (File.Exists(tempFile))
                File.Delete(tempFile);

            using (TokenTextWriter writer = new TokenTextWriter(tempFile, true))
                program.Emit(writer);
            return tempFile;
        }
    }
}
