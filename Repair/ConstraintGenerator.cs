using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

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
        /// <param name="errors">The errors.</param>
        /// <param name="assignments">The assignments provided by the solver.</param>
        public void ConstraintProgram(IEnumerable<Error> errors, Dictionary<string, bool> assignments)
        {
            IEnumerable<Implementation> implementations = errors.Select(x => x.Implementation).Distinct();
            foreach (Implementation implementation in implementations)
            {
                Procedure procedure = program.Procedures.Where(x => x.Name == implementation.Name).First();
                IEnumerable<Variable> variables = errors.Where(x => x.Implementation == implementation)
                    .SelectMany(x => x.Variables).Distinct();

                var pairs = from assignment in assignments
                            join variable in variables
                            on assignment.Key equals variable.Name
                            select new
                            {
                                Variable = variable,
                                Assignment = assignment.Value
                            };

                ApplyAssignments(procedure, pairs);
            }
        }

        /// <summary>
        /// Apply the assignments in the form of requires statements.
        /// </summary>
        /// <param name="procedure">The procdeure.</param>
        /// <param name="pairs">The key-value pairs representing the variable and it's assignment value.</param>
        private void ApplyAssignments(Procedure procedure, IEnumerable<dynamic> pairs)
        {
            foreach (dynamic pair in pairs)
            {
                if (pair.Assignment)
                {
                    procedure.Requires.Add(new Requires(false, new NAryExpr(Token.NoToken,
                        new BinaryOperator(Token.NoToken, BinaryOperator.Opcode.Eq), new List<Expr>()
                        {
                            new IdentifierExpr(Token.NoToken, pair.Variable),
                            new LiteralExpr(Token.NoToken, pair.Assignment)
                        })));
                }
            }
        }
    }
}