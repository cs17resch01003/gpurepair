using System.Collections.Generic;
using System.Text;

namespace GPURepair.Solvers
{
    public class Clause
    {
        /// <summary>
        /// The literals of the clause.
        /// </summary>
        public List<Literal> Literals { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        public Clause()
        {
            Literals = new List<Literal>();
        }

        /// <summary>
        /// Adds the literal to the clause.
        /// </summary>
        /// <param name="literal">The literal.</param>
        public void Add(Literal literal)
        {
            Literals.Add(literal);
        }

        /// <summary>
        /// Adds the literals to the clause.
        /// </summary>
        /// <param name="literals">The literals.</param>
        public void AddRange(IEnumerable<Literal> literals)
        {
            Literals.AddRange(literals);
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            foreach (Literal literal in Literals)
                builder.AppendFormat("{0}{1} ", literal.Value ? string.Empty : "-", literal.Variable);

            return builder.ToString();
        }
    }
}