using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Boogie;
using Microsoft.Boogie.GraphUtil;

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
            GenerateMetadata(program);
            PopulateDescendants();
        }

        /// <summary>
        /// Graphs generated from the source program.
        /// </summary>
        private Dictionary<Implementation, Graph<Block>> graphs;

        /// <summary>
        /// Nodes generated from the source program.
        /// </summary>
        private Dictionary<string, ProgramNode> nodes;

        /// <summary>
        /// Nodes generated from the source program.
        /// </summary>
        public IEnumerable<ProgramNode> Nodes => nodes.Values;

        /// <summary>
        /// The back edges in the source program.
        /// </summary>
        public List<BackEdge> BackEdges { get; private set; }

        /// <summary>
        /// Gets the node based on the implementation and the block.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        public ProgramNode GetNode(Implementation implementation, Block block)
        {
            string key = string.Format("{0}_{1}", implementation.Name, block.Label);
            if (nodes.ContainsKey(key))
                return nodes[key];
            return null;
        }

        /// <summary>
        /// Gets the loop nodes between the two edge points.
        /// </summary>
        /// <param name="edge">The back edge.</param>
        /// <returns>The loop nodes.</returns>
        public List<ProgramNode> GetLoopNodes(BackEdge edge)
        {
            Graph<Block> graph = graphs[edge.Source.Implementation];
            IEnumerable<Block> nodes = graph.NaturalLoops(edge.Destination.Block, edge.Source.Block);

            return nodes.Select(x => GetNode(edge.Source.Implementation, x)).ToList();
        }

        /// <summary>
        /// Generates the metadata from the program.
        /// </summary>
        private void GenerateMetadata(Program program)
        {
            graphs = new Dictionary<Implementation, Graph<Block>>();
            nodes = new Dictionary<string, ProgramNode>();

            BackEdges = new List<BackEdge>();

            foreach (Implementation implementation in program.Implementations)
            {
                Graph<Block> graph = Program.GraphFromImpl(implementation);
                graph.ComputeLoops();

                graphs.Add(implementation, graph);
                foreach (Block block in implementation.Blocks)
                    CreateNode(implementation, block);

                foreach (Block block in implementation.Blocks)
                {
                    ProgramNode node = GetNode(implementation, block);
                    IEnumerable<Block> successors, predecessors;

                    // Boogie throws KeyNotFoundException for unreachable code
                    try { successors = graph.Successors(node.Block); }
                    catch (Exception) { successors = new List<Block>(); }

                    try { predecessors = graph.Successors(node.Block); }
                    catch (Exception) { predecessors = new List<Block>(); }

                    node.Successors = successors.Select(x => GetNode(node.Implementation, x)).ToList();
                    node.Predecessors = predecessors.Select(x => GetNode(node.Implementation, x)).ToList();
                }

                foreach (Block header in graph.Headers)
                    foreach (Block backEdgeNode in graph.BackEdgeNodes(header))
                    {
                        ProgramNode source = GetNode(implementation, backEdgeNode);
                        ProgramNode destination = GetNode(implementation, header);

                        BackEdges.Add(new BackEdge(source, destination));
                    }
            }
        }

        /// <summary>
        /// Creates the node based on the implementation and the block.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        private void CreateNode(Implementation implementation, Block block)
        {
            if (GetNode(implementation, block) == null)
            {
                string key = string.Format("{0}_{1}", implementation.Name, block.Label);
                nodes.Add(key, new ProgramNode(implementation, block));
            }
        }

        /// <summary>
        /// Populates the descendants for the nodes.
        /// </summary>
        private void PopulateDescendants()
        {
            foreach (ProgramNode node in Nodes)
            {
                node.Descendants = new List<ProgramNode>();

                Queue<ProgramNode> queue = new Queue<ProgramNode>();
                queue.Enqueue(node);

                while (queue.Any())
                {
                    ProgramNode temp = queue.Dequeue();
                    if (node.Descendants.Contains(temp))
                        continue;

                    node.Descendants.Add(temp);
                    foreach (ProgramNode child in temp.Successors)
                        queue.Enqueue(child);
                }

                // remove the node from its descendants list
                node.Descendants.Remove(node);
            }
        }
    }
}
