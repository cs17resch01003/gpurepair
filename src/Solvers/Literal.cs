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
    }
}
