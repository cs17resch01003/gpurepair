using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Instrumentation
{
    public class BlockNode
    {
        private List<BlockNode> descendants = null;

        public BlockNode(Implementation implementation, Block block)
        {
            Implementation = implementation;
            Block = block;
            Label = block.Label;
        }

        public Implementation Implementation { get; set; }

        public Block Block { get; set; }

        public string Label { get; set; }

        public bool GlobalVariablePresent { get; set; } = false;

        public List<BlockNode> Parents { get; set; } = new List<BlockNode>();

        public List<BlockNode> Children { get; set; } = new List<BlockNode>();

        public List<BlockNode> GetDescendants()
        {
            if (descendants != null)
                return descendants;
            descendants = new List<BlockNode>();

            Queue<BlockNode> queue = new Queue<BlockNode>();
            queue.Enqueue(this);

            while (queue.Any())
            {
                BlockNode node = queue.Dequeue();
                foreach (BlockNode child in node.Children)
                {
                    if (!descendants.Contains(child))
                    {
                        descendants.Add(child);
                        queue.Enqueue(child);
                    }
                }
            }

            return descendants;
        }
    }
}
