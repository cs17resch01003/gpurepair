namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.Boogie;

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
        public Microsoft.Boogie.Program ConstraintProgram(Dictionary<string, bool> assignments)
        {
            Microsoft.Boogie.Program program = BoogieUtilities.ReadFile(filePath);
            if (assignments == null || !assignments.Any())
                return program;

            ApplyAssignments(program, assignments);

            string tempFile = BoogieUtilities.WriteFile(filePath, program);
            program = BoogieUtilities.ReadFile(tempFile);

            return program;
        }

        /// <summary>
        /// Apply the assignments in the form of axioms.
        /// </summary>
        /// <param name="program">The program.</param>
        /// <param name="assignments">The assignments provided by the solver.</param>
        private void ApplyAssignments(Microsoft.Boogie.Program program, Dictionary<string, bool> assignments)
        {
            // remove existing axioms
            List<Axiom> axioms = new List<Axiom>();
            foreach (Axiom axiom in program.Axioms.Where(x => x.Comment == "repair_constraint"))
                axioms.Add(axiom);

            axioms.ForEach(x => program.RemoveTopLevelDeclaration(x));

            // add the axioms corresponding to the assignments
            foreach (KeyValuePair<string, bool> assignment in assignments)
            {
                Expr identifier = new IdentifierExpr(Token.NoToken, assignment.Key, Type.Bool);
                Expr literal = new LiteralExpr(Token.NoToken, assignment.Value);

                BinaryOperator _operator = new BinaryOperator(Token.NoToken, BinaryOperator.Opcode.Iff);
                NAryExpr expr = new NAryExpr(Token.NoToken, _operator, new List<Expr> { identifier, literal });

                Axiom axiom = new Axiom(Token.NoToken, expr, "repair_constraint");
                program.AddTopLevelDeclaration(axiom);
            }
        }
    }
}
