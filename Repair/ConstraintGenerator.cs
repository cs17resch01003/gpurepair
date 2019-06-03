using Microsoft.Boogie;
using System.Collections.Generic;

namespace GPURepair.Repair
{
    public class ConstraintGenerator
    {
        /// <summary>
        /// The program to be instrumented.
        /// </summary>
        private Microsoft.Boogie.Program program;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public ConstraintGenerator(Microsoft.Boogie.Program program)
        {
            this.program = program;
        }

        /// <summary>
        /// Applies the required constraints to the program.
        /// </summary>
        /// <param name="assignments">The assignments provided by the solver.</param>
        public void ConstraintProgram(Dictionary<string, bool> assignments)
        {
            foreach (Implementation implementation in program.Implementations)
                if (ContainsAttribute(implementation, "source_name"))
                    ApplyAssignments(implementation, assignments);
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