using Microsoft.Boogie;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GPURepair.Repair
{
    public class ConstraintGenerator
    {
        /// <summary>
        /// The file path.
        /// </summary>
        private string filePath;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public ConstraintGenerator(string filePath)
        {
            this.filePath = filePath;
        }

        /// <summary>
        /// Applies the required constraints to the program.
        /// </summary>
        /// <param name="assignments">The assignments provided by the solver.</param>
        public Microsoft.Boogie.Program ConstraintProgram(Dictionary<string, bool> assignments)
        {
            Microsoft.Boogie.Program program = ReadFile(filePath);
            if (assignments == null || !assignments.Any())
                return program;


            foreach (Implementation implementation in program.Implementations)
                if (ContainsAttribute(implementation, "kernel"))
                    ApplyAssignments(implementation, assignments);

            string tempFile = WriteFile(program);
            program = ReadFile(tempFile);

            return program;
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
            if (File.Exists(tempFile))
                File.Delete(tempFile);

            using (TokenTextWriter writer = new TokenTextWriter(tempFile, true))
                program.Emit(writer);
            return tempFile;
        }

        /// <summary>
        /// Apply the assignments in the form of requires statements.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The assignments provided by the solver.</param>
        private void ApplyAssignments(Implementation implementation, Dictionary<string, bool> assignments)
        {
            foreach (KeyValuePair<string, bool> assignment in assignments)
            {
                SimpleAssignLhs assign = new SimpleAssignLhs(Token.NoToken, new IdentifierExpr(Token.NoToken, assignment.Key, Type.Bool));
                Expr literal = new LiteralExpr(Token.NoToken, assignment.Value);
                AssignCmd command = new AssignCmd(Token.NoToken, new List<AssignLhs> { assign },
                    new List<Expr> { literal });

                implementation.Blocks[0].Cmds.Insert(0, command);
            }
        }

        /// <summary>
        /// Checks if an implementation has the given attribute.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private bool ContainsAttribute(Implementation implementation, string attribute)
        {
            QKeyValue keyValue = implementation.Attributes;
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