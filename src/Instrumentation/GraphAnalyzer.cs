﻿using System.Collections.Generic;
using System.Linq;
using GPURepair.Common;
using Microsoft.Boogie;

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
        /// <param name="program">The boogie program.</param>
        public GraphAnalyzer(Microsoft.Boogie.Program program)
        {
            graph = new ProgramGraph(program);
        }

        /// <summary>
        /// Tracks that a global variable is present in the block.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        public void LinkBarrier(Implementation implementation, Block block)
        {
            ProgramNode node = graph.GetNode(implementation, block);
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
            foreach (ProgramNode node in graph.Nodes)
            {
                // identify merge points
                if (node.Predecessors.Count > 1)
                {
                    List<BackEdge> backEdges = graph.BackEdges;
                    if (backEdges.Any(x => x.Destination == node))
                    {
                        // loop structures
                        BackEdge edge = backEdges.First(x => x.Destination == node);
                        List<ProgramNode> loopNodes = graph.GetLoopNodes(edge);

                        if (loopNodes.Any(x => nodesContainingBarriers.Contains(x)))
                            list.Add(node);
                    }
                    else
                    {
                        // if-else diamond structures
                        ProgramNode lca = GetLCA(node.Predecessors);
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

                temp.Predecessors.ForEach(x => queue.Enqueue(x));
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
                if (!node.Successors.Contains(end))
                {
                    foreach (ProgramNode child in node.Successors)
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
