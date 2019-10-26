using System.Collections.Generic;
using System.IO;
using System.Linq;
using GPURepair.Repair.Errors;
using GPURepair.Repair.Metadata;
using Microsoft.Boogie;

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
        /// <param name="errors">The errors.</param>
        public Microsoft.Boogie.Program ConstraintProgram(
            Dictionary<string, bool> assignments, IEnumerable<RepairableError> errors)
        {
            Microsoft.Boogie.Program program = BoogieUtilities.ReadFile(filePath);
            if (assignments == null || !assignments.Any())
                return program;

            foreach (Implementation implementation in program.Implementations)
            {
                Dictionary<string, bool> current_assignments = new Dictionary<string, bool>();
                IEnumerable<string> barrierNames = ProgramMetadata.Barriers
                    .Where(x => x.Value.Implementation.Name == implementation.Name).Select(x => x.Value.Name);

                foreach (string barrierName in barrierNames)
                    if (assignments.ContainsKey(barrierName))
                        current_assignments.Add(barrierName, assignments[barrierName]);

                if (current_assignments.Any())
                    ApplyAssignments(implementation, current_assignments);
            }

            string tempFile = WriteFile(program);
            program = BoogieUtilities.ReadFile(tempFile);

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
    }
}
