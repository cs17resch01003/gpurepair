namespace GPURepair.Solvers
{
    using System.Collections.Generic;
    using System.Linq;

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
            return string.Join(" ", Literals.Distinct().OrderBy(x => x.Value)
                .ThenBy(x => int.Parse(x.Variable.Replace("b", string.Empty))).Select(x => x.ToString()));
        }

        /// <summary>
        /// Determines if the specified object is equal to the current object.
        /// </summary>
        /// <param name="obj">The specified object.</param>
        /// <returns>true if the specified object is equal to the current object; otherwise false.</returns>
        public override bool Equals(object obj)
        {
            Clause clause = obj as Clause;
            if (clause == null)
                return false;

            return clause.ToString().Equals(ToString());
        }

        /// <summary>
        /// Serves as the default hash function.
        /// </summary>
        /// <returns>A hash code for the current object.</returns>
        public override int GetHashCode()
        {
            return ToString().GetHashCode();
        }
    }
}
