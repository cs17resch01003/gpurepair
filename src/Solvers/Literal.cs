namespace GPURepair.Solvers
{
    public class Literal
    {
        /// <summary>
        /// The variable name.
        /// </summary>
        public string Variable { get; set; }

        /// <summary>
        /// The polarity of the literal.
        /// </summary>
        public bool Value { get; set; }

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="variable">The variable name.</param>
        /// <param name="value">The polarity of the literal.</param>
        public Literal(string variable, bool value)
        {
            Variable = variable;
            Value = value;
        }

        /// <summary>
        /// Returns a string that represents the current object.
        /// </summary>
        /// <returns>The string representation.</returns>
        public override string ToString()
        {
            return string.Format("{0}{1}", Value ? string.Empty : "-", Variable);
        }

        /// <summary>
        /// Determines if the specified object is equal to the current object.
        /// </summary>
        /// <param name="obj">The specified object.</param>
        /// <returns>true if the specified object is equal to the current object; otherwise false.</returns>
        public override bool Equals(object obj)
        {
            Literal literal = obj as Literal;
            if (literal == null)
                return false;

            return literal.Variable == Variable && literal.Value == Value;
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
