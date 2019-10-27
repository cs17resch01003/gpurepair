using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Instrumentation
{
    public static class BlockNodeManager
    {
        /// <summary>
        /// Nodes generated from the source program.
        /// </summary>
        private static Dictionary<string, BlockNode> nodes = new Dictionary<string, BlockNode>();

        /// <summary>
        /// Generates the nodes.
        /// </summary>
        public static void GenerateNodes(Microsoft.Boogie.Program program)
        {
            foreach (Implementation implementation in program.Implementations)
                foreach (Block block in implementation.Blocks)
                    CreateNode(implementation, block);

            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    BlockNode parent = GetNode(implementation.Name, block.Label);

                    GotoCmd command = block.TransferCmd as GotoCmd;
                    if (command != null)
                    {
                        foreach (string label in command.labelNames)
                        {
                            BlockNode child = GetNode(implementation.Name, label);
                            parent.Children.Add(child);
                            child.Parents.Add(parent);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Sets the flag that a global variable is present in the block.
        /// </summary>
        /// <param name="implementation">The implementation name.</param>
        /// <param name="block">The block label.</param>
        public static void SetGlobalVariablePresent(string implementation, string block)
        {
            string key = string.Format("{0}_{1}", implementation, block);
            if (nodes.ContainsKey(key))
                nodes[key].GlobalVariablePresent = true;
        }

        /// <summary>
        /// Gets the nodes where the code graph is merging.
        /// </summary>
        /// <returns>The merge nodes.</returns>
        public static List<BlockNode> GetMergeNodes()
        {
            List<BlockNode> list = new List<BlockNode>();

            // look for diamond structures and loops in the code
            // this is useful for fixing errors where the usage of the global variable is inside the if/else block
            // but the solution needs a barrier outside the block
            foreach (BlockNode node in nodes.Values)
            {
                // identify merge points
                if (node.Parents.Count > 1)
                {
                    if (node.GetDescendants().Contains(node))
                    {
                        // loop structures
                        if (node.GetDescendants().Any(x => x.GlobalVariablePresent))
                            list.Add(node);
                    }
                    else
                    {
                        // if-else diamond structures
                        BlockNode lca = GetLCA(node.Parents);
                        if (lca != null)
                        {
                            List<BlockNode> intermediateNodes = GetIntermediateNodes(lca, node);
                            if (intermediateNodes.Any(x => x.GlobalVariablePresent))
                                list.Add(node);
                        }
                    }
                }
            }

            return list;
        }

        /// <summary>
        /// Creates the node based on the implementation name and the block label.
        /// </summary>
        /// <param name="implementation">The implementation.</param>
        /// <param name="block">The block.</param>
        private static void CreateNode(Implementation implementation, Block block)
        {
            string key = string.Format("{0}_{1}", implementation.Name, block.Label);
            if (!nodes.ContainsKey(key))
                nodes.Add(key, new BlockNode(implementation, block));
        }

        /// <summary>
        /// Retrieves the node based on the implementation name and the block label.
        /// </summary>
        /// <param name="implementation">The implementation name.</param>
        /// <param name="block">The block label.</param>
        /// <returns>The node.</returns>
        private static BlockNode GetNode(string implementation, string block)
        {
            string key = string.Format("{0}_{1}", implementation, block);
            return nodes[key];
        }

        /// <summary>
        /// Identifies the least common ancestor for the given nodes.
        /// </summary>
        /// <param name="nodes">The nodes.</param>
        /// <returns>The LCA.</returns>
        private static BlockNode GetLCA(List<BlockNode> nodes)
        {
            // traverse up one of the parents until the LCA is found
            // it's enough to traverse only one path upwards since we will always find the LCA on this path
            Queue<BlockNode> queue = new Queue<BlockNode>();
            queue.Enqueue(nodes.First());

            while (queue.Any())
            {
                BlockNode temp = queue.Dequeue();
                if (ContainsNodes(temp.GetDescendants(), nodes))
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
        private static bool ContainsNodes(List<BlockNode> list, List<BlockNode> searchNodes)
        {
            foreach (BlockNode searchNode in searchNodes)
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
        private static List<BlockNode> GetIntermediateNodes(BlockNode start, BlockNode end)
        {
            Queue<BlockNode> queue = new Queue<BlockNode>();
            queue.Enqueue(start);

            List<BlockNode> nodes = new List<BlockNode>();
            while (queue.Any())
            {
                BlockNode node = queue.Dequeue();
                if (!node.Children.Contains(end))
                {
                    foreach (BlockNode child in node.Children)
                    {
                        // skip paths which do not have the end node in them
                        if (!nodes.Contains(child) && child.GetDescendants().Contains(end))
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
