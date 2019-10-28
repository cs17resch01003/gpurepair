using System.Collections.Generic;
using System.Linq;
using Microsoft.Boogie;

namespace GPURepair.Common
{
    /// <summary>
    /// Builds a graph from the given Boogie program.
    /// </summary>
    public class ProgramGraph
    {
        /// <summary>
        /// Initilaizes an instance of <see cref="ProgramGraph"/>.
        /// </summary>
        /// <param name="program">The program.</param>
        public ProgramGraph(Program program)
        {
            GenerateNodes(program);
            TraverseGraph();
        }

        /// <summary>
        /// Nodes generated from the source program.
        /// </summary>
        private Dictionary<string, ProgramNode> nodes = new Dictionary<string, ProgramNode>();

        /// <summary>
        /// The back edges in the source program.
        /// </summary>
        private List<(ProgramNode, ProgramNode)> backEdges = new List<(ProgramNode, ProgramNode)>();

        /// <summary>
        /// Gets the nodes in the program graph.
        /// </summary>
        /// <returns>The program nodes.</returns>
        public List<ProgramNode> GetNodes()
        {
            return nodes.Values.ToList();
        }

        /// <summary>
        /// Gets the back edges in the program graph.
        /// </summary>
        /// <returns>The back edges.</returns>
        public List<(ProgramNode, ProgramNode)> GetBackEdges()
        {
            return backEdges;
        }

        /// <summary>
        /// Gets the node based on the implementation name and block label.
        /// </summary>
        /// <param name="implementationName">The implementation name.</param>
        /// <param name="blockLabel">The block label.</param>
        /// <returns>The program node.</returns>
        public ProgramNode GetNode(string implementationName, string blockLabel)
        {
            string key = string.Format("{0}_{1}", implementationName, blockLabel);
            if (nodes.ContainsKey(key))
                return nodes[key];

            return null;
        }

        /// <summary>
        /// Generates the nodes.
        /// </summary>
        private void GenerateNodes(Program program)
        {
            foreach (Implementation implementation in program.Implementations)
                foreach (Block block in implementation.Blocks)
                    CreateNode(implementation, block);

            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    ProgramNode parent = GetNode(implementation.Name, block.Label);
                    GotoCmd command = block.TransferCmd as GotoCmd;

                    if (command != null)
                    {
                        foreach (string label in command.labelNames)
                        {
                            ProgramNode child = GetNode(implementation.Name, label);
                            parent.Children.Add(child);
                            child.Parents.Add(parent);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Creates the node based on the implementation name and the block label.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        private void CreateNode(Implementation implementation, Block block)
        {
            string key = string.Format("{0}_{1}", implementation.Name, block.Label);
            if (!nodes.ContainsKey(key))
                nodes.Add(key, new ProgramNode(implementation, block));
        }

        /// <summary>
        /// Runs a graph traversal algorithm and populates the descendants.
        /// Also identifies the back-edges in the graph.
        /// </summary>
        private void TraverseGraph()
        {
            IEnumerable<ProgramNode> entryNodes = nodes.Values.Where(x => !x.Parents.Any());
            foreach (ProgramNode entryNode in entryNodes)
                TraverseGraph(entryNode, new List<ProgramNode>());
        }

        /// <summary>
        /// The recursive traversal logic.
        /// </summary>
        /// <param name="node">The node to be evaluated.</param>
        /// <param name="path">The grpah path so far.</param>
        private void TraverseGraph(ProgramNode node, List<ProgramNode> path)
        {
            foreach (ProgramNode child in node.Children)
            {
                if (path.Contains(child))
                    AddBackEdge(node, child);
                else
                {
                    List<ProgramNode> new_path = new List<ProgramNode>(path);
                    new_path.Add(node);

                    foreach (ProgramNode pathNode in new_path)
                        if (!pathNode.Descendants.Contains(child))
                            pathNode.Descendants.Add(child);

                    TraverseGraph(child, new_path);
                }
            }
        }

        /// <summary>
        /// Adds a back edge with the given nodes.
        /// </summary>
        /// <param name="source">The source of the back edge.</param>
        /// <param name="destination">The destination of the back edge.</param>
        private void AddBackEdge(ProgramNode source, ProgramNode destination)
        {
            if (!backEdges.Any(x => x.Item1 == source && x.Item2 == destination))
                backEdges.Add((source, destination));
        }
    }
}
