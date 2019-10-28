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
        public Implementation Implementation { get; set; }

        /// <summary>
        /// The block.
        /// </summary>
        public Block Block { get; set; }

        /// <summary>
        /// The parents of the node.
        /// </summary>
        public List<ProgramNode> Parents { get; } = new List<ProgramNode>();

        /// <summary>
        /// The children of the node.
        /// </summary>
        public List<ProgramNode> Children { get; } = new List<ProgramNode>();

        /// <summary>
        /// The descendants of the node.
        /// </summary>
        public List<ProgramNode> Descendants { get; } = new List<ProgramNode>();
    }
}
