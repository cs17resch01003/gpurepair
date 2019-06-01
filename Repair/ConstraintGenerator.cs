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
        /// <param name="assignments">The assignments provided by the solver.</param>
        public void ConstraintProgram(Dictionary<string, bool> assignments)
        {
            Dictionary<string, Axiom> axioms = new Dictionary<string, Axiom>();
            foreach (Axiom axiom in program.Axioms)
            {
                if (ContainsAttribute(axiom, "repair"))
                {
                    NAryExpr expression = axiom.Expr as NAryExpr;
                    IdentifierExpr identifier = expression.Args.Where(x => x is IdentifierExpr)
                        .Select(x => x as IdentifierExpr).First();

                    axioms.Add(identifier.Name, axiom);
                }
            }

            foreach (string variable in assignments.Keys)
            {
                IdentifierExpr identifier = new IdentifierExpr(Token.NoToken, variable, Type.Bool);
                LiteralExpr literal = new LiteralExpr(Token.NoToken, assignments[variable]);

                IAppliable function = new BinaryOperator(Token.NoToken, BinaryOperator.Opcode.Eq);
                NAryExpr expression = new NAryExpr(Token.NoToken, function,
                    new List<Expr>() { identifier, literal });

                if (axioms.ContainsKey(variable))
                {
                    Axiom axiom = axioms[variable];
                    axiom.Expr = expression;
                }
                else
                {
                    Axiom axiom = new Axiom(Token.NoToken, expression);

                    Dictionary<string, object> attributes = new Dictionary<string, object>();
                    attributes.Add("repair", null);

                    axiom.Attributes = ConvertAttributes(attributes);
                    program.AddTopLevelDeclaration(axiom);
                }
            }
        }

        /// <summary>
        /// Checks if an axiom has the given attribute.
        /// </summary>
        /// <param name="axiom">The axiom.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private bool ContainsAttribute(Axiom axiom, string attribute)
        {
            QKeyValue keyValue = axiom.Attributes;
            bool found = false;

            while (keyValue != null && !found)
            {
                if (attribute == keyValue.Key)
                    found = true;
                keyValue = keyValue.Next;
            }

            return found;
        }

        /// <summary>
        /// Generates a key-value chain from the given attributes.
        /// </summary>
        /// <param name="attributes">The attributes.</param>
        /// <returns>The first node in the attribute chain.</returns>
        private QKeyValue ConvertAttributes(Dictionary<string, object> attributes)
        {
            QKeyValue value = null;
            foreach (KeyValuePair<string, object> pair in attributes)
            {
                List<object> array = new List<object>();
                if (pair.Value != null)
                    array.Add(pair.Value);

                QKeyValue temp = new QKeyValue(Token.NoToken, pair.Key, array, value);
                value = temp;
            }

            return value;
        }
    }
}