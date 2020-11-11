namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Repair.Metadata;
    using Microsoft.Boogie;

    public class ConstraintGenerator
    {
        /// <summary>
        /// The file path.
        /// </summary>
        private string filePath;

        /// <summary>
        /// Use axioms for instrumentation instead of variable assignments.
        /// </summary>
        private bool useAxioms;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="useAxioms">Use axioms for instrumentation instead of variable assignments.</param>
        public ConstraintGenerator(string filePath, bool useAxioms)
        {
            this.filePath = filePath;
            this.useAxioms = useAxioms;
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

            if (useAxioms)
            {
                ApplyAssignments(program, assignments);
            }
            else
            {
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
            }

            string tempFile = BoogieUtilities.WriteFile(filePath, program);
            program = BoogieUtilities.ReadFile(tempFile);

            return program;
        }

        /// <summary>
        /// Apply the assignments in the form of assign statements.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="assignments">The assignments provided by the solver.</param>
        private void ApplyAssignments(Implementation implementation, Dictionary<string, bool> assignments)
        {
            foreach (KeyValuePair<string, bool> assignment in assignments)
            {
                IdentifierExpr expr = new IdentifierExpr(Token.NoToken, assignment.Key, Type.Bool);
                Expr literal = new LiteralExpr(Token.NoToken, assignment.Value);

                SimpleAssignLhs assign = new SimpleAssignLhs(Token.NoToken, expr);
                AssignCmd command = new AssignCmd(Token.NoToken, new List<AssignLhs> { assign },
                    new List<Expr> { literal });

                implementation.Blocks[0].Cmds.Insert(0, command);
            }
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
