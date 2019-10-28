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
            PopulateLoopNodes();
        }

        /// <summary>
        /// Nodes generated from the source program.
        /// </summary>
        private Dictionary<string, ProgramNode> nodes = new Dictionary<string, ProgramNode>();

        /// <summary>
        /// The back edges in the source program.
        /// </summary>
        private List<BackEdge> backEdges = new List<BackEdge>();

        /// <summary>
        /// The back edge and the corresponding nodes in the loop.
        /// </summary>
        private Dictionary<BackEdge, List<ProgramNode>> loopNodes = new Dictionary<BackEdge, List<ProgramNode>>();

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
        public List<BackEdge> GetBackEdges()
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
        /// Gets the loop nodes between the two edge points.
        /// </summary>
        /// <param name="edge">The back-edge.</param>
        /// <returns>The loop nodes.</returns>
        public List<ProgramNode> GetLoopNodes(BackEdge edge)
        {
            return loopNodes[edge];
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
            if (!backEdges.Any(x => x.Source == source && x.Destination == destination))
                backEdges.Add(new BackEdge(source, destination));
        }

        /// <summary>
        /// Popluates the nodes in the path of each back edge.
        /// </summary>
        private void PopulateLoopNodes()
        {
            Dictionary<BackEdge, List<ProgramNode>> pathNodes = new Dictionary<BackEdge, List<ProgramNode>>();
            foreach (BackEdge edge in backEdges)
            {
                List<ProgramNode> nodes = GeneratePathNodes(edge);
                pathNodes.Add(edge, nodes);
            }

            List<(BackEdge, BackEdge)> relatedEdges = new List<(BackEdge, BackEdge)>();
            foreach (BackEdge edge1 in backEdges)
                foreach (BackEdge edge2 in backEdges)
                    if (edge1 != edge2)
                    {
                        bool first = pathNodes[edge2].Contains(edge1.Source) || pathNodes[edge2].Contains(edge1.Destination);
                        bool second = pathNodes[edge1].Contains(edge2.Source) || pathNodes[edge1].Contains(edge2.Destination);

                        if (first || second)
                            relatedEdges.Add((edge1, edge2));
                    }

            foreach (BackEdge edge in backEdges)
            {
                List<BackEdge> related = GetRelatedEdges(relatedEdges, edge);
                List<ProgramNode> nodes = pathNodes.Where(x => related.Contains(x.Key)).SelectMany(x => x.Value).Distinct().ToList();

                loopNodes.Add(edge, nodes);
            }
        }

        /// <summary>
        /// Gets the related edges recursively.
        /// </summary>
        /// <param name="relatedEdges">The first level related edges.</param>
        /// <param name="edge">The edge for which the related edges are determined.</param>
        /// <returns></returns>
        private List<BackEdge> GetRelatedEdges(List<(BackEdge, BackEdge)> relatedEdges, BackEdge edge)
        {
            List<BackEdge> result = new List<BackEdge>();

            Queue<BackEdge> queue = new Queue<BackEdge>();
            queue.Enqueue(edge);

            while (queue.Any())
            {
                BackEdge temp = queue.Dequeue();
                result.Add(temp);

                IEnumerable<BackEdge> first = relatedEdges.Where(x => x.Item1 == temp).Select(x => x.Item2);
                IEnumerable<BackEdge> second = relatedEdges.Where(x => x.Item2 == temp).Select(x => x.Item1);

                foreach (BackEdge related in first.Union(second).Distinct())
                    if (!result.Contains(related))
                        queue.Enqueue(related);
            }

            return result;
        }

        /// <summary>
        /// Gets the nodes in the path between the two edge points.
        /// </summary>
        /// <param name="edge">The back-edge.</param>
        /// <returns>The path nodes.</returns>
        private List<ProgramNode> GeneratePathNodes(BackEdge edge)
        {
            List<ProgramNode> pathNodes = new List<ProgramNode>();
            ProgramNode mergeNode = edge.Destination;

            Queue<ProgramNode> queue = new Queue<ProgramNode>();
            queue.Enqueue(mergeNode);

            while (queue.Any())
            {
                ProgramNode node = queue.Dequeue();
                if (!pathNodes.Contains(node) && (node == edge.Source || node.Descendants.Contains(edge.Source)))
                {
                    pathNodes.Add(node);
                    node.Children.ForEach(x => queue.Enqueue(x));
                }
            }

            return pathNodes;
        }
    }
}
