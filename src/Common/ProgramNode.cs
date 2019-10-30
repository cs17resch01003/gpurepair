using System.Collections.Generic;
using Microsoft.Boogie;

namespace GPURepair.Common
{
    /// <summary>
    /// Represents a node in the program graph.
    /// </summary>
    public class ProgramNode
    {
        /// <summary>
        /// Initilaizes an instance of <see cref="ProgramNode"/>.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        public ProgramNode(Implementation implementation, Block block)
        {
            Implementation = implementation;
            Block = block;
        }

        /// <summary>
        /// The implementation.
        /// </summary>
        public Implementation Implementation { get; private set; }

        /// <summary>
        /// The block.
        /// </summary>
        public Block Block { get; private set; }

        /// <summary>
        /// The successors of the node.
        /// </summary>
        public List<ProgramNode> Successors { get; internal set; }

        /// <summary>
        /// The predecessors of the node.
        /// </summary>
        public List<ProgramNode> Predecessors { get; internal set; }

        /// <summary>
        /// The descendants of the node ignoring back edges.
        /// </summary>
        public List<ProgramNode> Descendants { get; internal set; }
    }
}
