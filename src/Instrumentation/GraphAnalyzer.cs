using System.Collections.Generic;
using System.Linq;
using GPURepair.Common;

namespace GPURepair.Instrumentation
{
    /// <summary>
    /// Analyzes the graph.
    /// </summary>
    public class GraphAnalyzer
    {
        /// <summary>
        /// The graph obtained the program.
        /// </summary>
        private ProgramGraph graph;

        /// <summary>
        /// Track the nodes which have barriers.
        /// </summary>
        private List<ProgramNode> nodesContainingBarriers = new List<ProgramNode>();

        /// <summary>
        /// Initilaizes an instance of <see cref="GraphAnalyzer"/>.
        /// </summary>
        /// <param name="program"></param>
        public GraphAnalyzer(Microsoft.Boogie.Program program)
        {
            graph = new ProgramGraph(program);
        }

        /// <summary>
        /// Sets the flag that a global variable is present in the block.
        /// </summary>
        /// <param name="implementation">The implementation name.</param>
        /// <param name="block">The block label.</param>
        public void LinkBarrier(string implementationName, string blockLabel)
        {
            ProgramNode node = graph.GetNode(implementationName, blockLabel);
            if (node != null && !nodesContainingBarriers.Contains(node))
                nodesContainingBarriers.Add(node);
        }

        /// <summary>
        /// Gets the nodes where the code graph is merging.
        /// </summary>
        /// <returns>The merge nodes.</returns>
        public List<ProgramNode> GetMergeNodes()
        {
            List<ProgramNode> list = new List<ProgramNode>();

            // look for diamond structures and loops in the code
            // this is useful for fixing errors where the usage of the global variable is inside the if/else block
            // but the solution needs a barrier outside the block
            foreach (ProgramNode node in graph.GetNodes())
            {
                // identify merge points
                if (node.Parents.Count > 1)
                {
                    List<(ProgramNode, ProgramNode)> backEdges = graph.GetBackEdges();
                    if (backEdges.Any(x => x.Item2 == node))
                    {
                        // loop structures
                        (ProgramNode, ProgramNode) edge = backEdges.First(x => x.Item2 == node);
                        List<ProgramNode> loopNodes = GetLoopNodes(edge);

                        if (loopNodes.Any(x => nodesContainingBarriers.Contains(x)))
                            list.Add(node);
                    }
                    else
                    {
                        // if-else diamond structures
                        ProgramNode lca = GetLCA(node.Parents);
                        if (lca != null)
                        {
                            List<ProgramNode> intermediateNodes = GetIntermediateNodes(lca, node);
                            if (intermediateNodes.Any(x => nodesContainingBarriers.Contains(x)))
                                list.Add(node);
                        }
                    }
                }
            }

            return list;
        }

        /// <summary>
        /// Gets the loop nodes between the two edge points.
        /// </summary>
        /// <param name="edge">The back-edge.</param>
        /// <returns>The loop nodes.</returns>
        private static List<ProgramNode> GetLoopNodes((ProgramNode, ProgramNode) edge)
        {
            List<ProgramNode> loopNodes = new List<ProgramNode>();

            ProgramNode mergeNode = edge.Item2;
            foreach (ProgramNode child in mergeNode.Children)
                if (child.Descendants.Contains(edge.Item1))
                {
                    loopNodes.Add(child);
                    loopNodes.AddRange(child.Descendants);
                }

            loopNodes.Add(mergeNode);
            return loopNodes;
        }

        /// <summary>
        /// Identifies the least common ancestor for the given nodes.
        /// </summary>
        /// <param name="nodes">The nodes.</param>
        /// <returns>The LCA.</returns>
        private static ProgramNode GetLCA(List<ProgramNode> nodes)
        {
            // traverse up one of the parents until the LCA is found
            // it's enough to traverse only one path upwards since we will always find the LCA on this path
            Queue<ProgramNode> queue = new Queue<ProgramNode>();
            queue.Enqueue(nodes.First());

            while (queue.Any())
            {
                ProgramNode temp = queue.Dequeue();
                if (ContainsNodes(temp.Descendants, nodes))
                    return temp;

                temp.Parents.ForEach(x => queue.Enqueue(x));
            }

            return null;
        }

        /// <summary>
        /// Checks if one node list is a part of the other list.
        /// </summary>
        /// <param name="list">The list to search in.</param>
        /// <param name="searchNodes">The list to search.</param>
        /// <returns>True if the nodes exists, False otherwise.</returns>
        private static bool ContainsNodes(List<ProgramNode> list, List<ProgramNode> searchNodes)
        {
            foreach (ProgramNode searchNode in searchNodes)
            {
                bool result = list.Contains(searchNode);
                if (result == false)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets the nodes between the given two nodes.
        /// </summary>
        /// <param name="start">The start node.</param>
        /// <param name="end">The end node.</param>
        /// <returns>The nodes between the given two nodes.</returns>
        private static List<ProgramNode> GetIntermediateNodes(ProgramNode start, ProgramNode end)
        {
            Queue<ProgramNode> queue = new Queue<ProgramNode>();
            queue.Enqueue(start);

            List<ProgramNode> nodes = new List<ProgramNode>();
            while (queue.Any())
            {
                ProgramNode node = queue.Dequeue();
                if (!node.Children.Contains(end))
                {
                    foreach (ProgramNode child in node.Children)
                    {
                        // skip paths which do not have the end node in them
                        if (!nodes.Contains(child) && child.Descendants.Contains(end))
                        {
                            nodes.Add(child);
                            queue.Enqueue(child);
                        }
                    }
                }
            }

            return nodes;
        }
    }
}
