namespace GPURepair.Solvers
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Solvers.Exceptions;
    using Microsoft.Z3;

    internal class MHSSolution
    {
        /// <summary>
        /// Lookup to get the clause state for the clause.
        /// </summary>
        private readonly Dictionary<Clause, State> clauseLookup;

        /// <summary>
        /// Lookup to get the clauses that the variable is a part of.
        /// </summary>
        internal readonly Dictionary<string, List<Clause>> VariableLookup;

        /// <summary>
        /// The assignments.
        /// </summary>
        internal readonly Dictionary<string, bool> Assignments;

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        public MHSSolution(IEnumerable<Clause> clauses)
        {
            clauseLookup = new Dictionary<Clause, State>();

            VariableLookup = new Dictionary<string, List<Clause>>();
            Assignments = new Dictionary<string, bool>();

            foreach (Clause clause in clauses)
            {
                State state = new State(clause);
                foreach (string variable in state.VariableLookup.Keys)
                {
                    if (!VariableLookup.ContainsKey(variable))
                        VariableLookup.Add(variable, new List<Clause>());
                    VariableLookup[variable].Add(clause);
                }

                clauseLookup.Add(clause, state);
            }
        }

        /// <summary>
        /// Sets the assignment for a variable.
        /// </summary>
        public void SetAssignment(string variable, bool value)
        {
            Assignments.Add(variable, value);
            foreach (Clause clause in VariableLookup[variable])
            {
                State state = clauseLookup[clause];
                state.UnassignedLiterals--;

                if (state.Polarity == value)
                    state.Sat = Status.SATISFIABLE;
            }

            // throw an error if this assignment causes a clause to go unsat
            IEnumerable<Clause> unsat_clauses = clauseLookup
                .Where(x => x.Value.Sat == Status.UNKNOWN && x.Value.UnassignedLiterals == 0).Select(x => x.Key);
            if (unsat_clauses.Any())
                throw new SatisfiabilityError();
        }

        /// <summary>
        /// Classifies the active clauses as positive and negative.
        /// </summary>
        /// <returns>The unit clauses.</returns>
        public List<Clause> GetActiveUnitClauses()
        {
            return clauseLookup.Where(x => x.Value.Sat == Status.UNKNOWN && x.Value.UnassignedLiterals == 1)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Returns all the active positive clauses.
        /// </summary>
        /// <returns>The positive clauses.</returns>
        public List<Clause> GetActivePositiveClauses()
        {
            return clauseLookup.Where(x => x.Value.Sat == Status.UNKNOWN && x.Value.Polarity == true)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Returns all the active positive clauses.
        /// </summary>
        /// <returns>The negative clauses.</returns>
        public List<Clause> GetActiveNegativeClauses()
        {
            return clauseLookup.Where(x => x.Value.Sat == Status.UNKNOWN && x.Value.Polarity == false)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Verifies if the solution is found.
        /// </summary>
        /// <returns>true if the solution is found, false otherwise.</returns>
        public bool IsSolutionFound()
        {
            return !clauseLookup.Any(x => x.Value.Sat == Status.UNKNOWN);
        }

        /// <summary>
        /// Represents the state of the clause.
        /// </summary>
        private class State
        {
            /// <summary>
            /// Indicates whether the literals are positive or negative.
            /// </summary>
            internal readonly bool Polarity;

            /// <summary>
            /// Lookup to get the literal based on the variable.
            /// </summary>
            internal readonly Dictionary<string, Literal> VariableLookup;

            /// <summary>
            /// Indicates whether the clause has been satisfied.
            /// </summary>
            internal Status Sat { get; set; } = Status.UNKNOWN;

            /// <summary>
            /// The number of unsat literals.
            /// </summary>
            internal ushort UnassignedLiterals { get; set; }

            /// <summary>
            /// The constructor.
            /// </summary>
            internal State(Clause clause)
            {
                Polarity = clause.Literals.First().Value;
                VariableLookup = new Dictionary<string, Literal>();

                foreach (Literal literal in clause.Literals)
                {
                    VariableLookup.Add(literal.Variable, literal);
                    UnassignedLiterals++;
                }
            }
        }
    }
}
