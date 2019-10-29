using System.Collections.Generic;
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
                        // can be simplified by identifying the least common ancestor and inspecting if the intermediate nodes have barriers
                        list.Add(node);
                    }
                }
            }

            return list;
        }
    }
}
