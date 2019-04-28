using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Solvers
{
    public class MHSSolution
    {
        /// <summary>
        /// The clauses.
        /// </summary>
        public readonly Dictionary<string, List<Clause>> VariableLookup = new Dictionary<string, List<Clause>>();

        /// <summary>
        /// The clauses.
        /// </summary>
        public readonly Dictionary<Clause, State> ClauseLookup = new Dictionary<Clause, State>();

        /// <summary>
        /// The assignments.
        /// </summary>
        public readonly Dictionary<string, bool> Assignments = new Dictionary<string, bool>();

        /// <summary>
        /// The constructor.
        /// </summary>
        /// <param name="clauses">The clauses.</param>
        public MHSSolution(IEnumerable<Clause> clauses)
        {
            foreach (Clause clause in clauses)
            {
                State state = new State(clause);
                foreach (string variable in state.VariableLookup.Keys)
                {
                    if (!VariableLookup.ContainsKey(variable))
                        VariableLookup.Add(variable, new List<Clause>());
                    VariableLookup[variable].Add(clause);
                }

                ClauseLookup.Add(clause, state);
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
                State state = ClauseLookup[clause];
                state.UnassignedLiterals--;

                if (state.Polarity == value)
                    state.Sat = true;
            }
        }

        /// <summary>
        /// Classifies the active clauses as positive and negative.
        /// </summary>
        /// <returns>The unit clauses.</returns>
        public List<Clause> GetActiveUnitClauses()
        {
            return ClauseLookup.Where(x => x.Value.Sat == false && x.Value.UnassignedLiterals == 1)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Returns all the active positive clauses.
        /// </summary>
        /// <returns>The positive clauses.</returns>
        public List<Clause> GetActivePositiveClauses()
        {
            return ClauseLookup.Where(x => x.Value.Sat == false && x.Value.Polarity == true)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Returns all the active positive clauses.
        /// </summary>
        /// <returns>The negative clauses.</returns>
        public List<Clause> GetActiveNegativeClauses()
        {
            return ClauseLookup.Where(x => x.Value.Sat == false && x.Value.Polarity == false)
                .Select(x => x.Key).ToList();
        }

        /// <summary>
        /// Verifies if the solution is found.
        /// </summary>
        /// <returns>true if the solution is found, false otherwise.</returns>
        public bool IsSolutionFound()
        {
            return !ClauseLookup.Any(x => x.Value.Sat == false);
        }

        /// <summary>
        /// Represents the state of the clause.
        /// </summary>
        public class State
        {
            /// <summary>
            /// Indicates whether the literals are positive or negative.
            /// </summary>
            public readonly bool Polarity;

            /// <summary>
            /// Lookup to get the literal based on the variable.
            /// </summary>
            public readonly Dictionary<string, Literal> VariableLookup = new Dictionary<string, Literal>();

            /// <summary>
            /// Indicates whether the clause has been satisfied.
            /// </summary>
            public bool Sat { get; set; }

            /// <summary>
            /// The number of unsat literals.
            /// </summary>
            public ushort UnassignedLiterals { get; set; }

            /// <summary>
            /// The constructor.
            /// </summary>
            public State(Clause clause)
            {
                Polarity = clause.Literals.First().Value;
                foreach (Literal literal in clause.Literals)
                {
                    VariableLookup.Add(literal.Variable, literal);
                    UnassignedLiterals++;
                }
            }
        }
    }
}