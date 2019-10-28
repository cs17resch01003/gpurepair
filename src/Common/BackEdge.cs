namespace GPURepair.Common
{
    /// <summary>
    /// Represents a back edge.
    /// </summary>
    public class BackEdge
    {
        /// <summary>
        /// Initilaizes an instance of <see cref="BackEdge"/>.
        /// </summary>
        /// <param name="source">The source node of the back edge.</param>
        /// <param name="destination">The destination node of the back edge.</param>
        public BackEdge(ProgramNode source, ProgramNode destination)
        {
            Source = source;
            Destination = destination;
        }

        /// <summary>
        /// The source node of the back edge.
        /// </summary>
        public ProgramNode Source { get; set; }

        /// <summary>
        /// The destination node of the back edge.
        /// </summary>
        public ProgramNode Destination { get; set; }
    }
}
