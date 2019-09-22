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

            descendants = new List<BlockNode>(Children);
            foreach (BlockNode child in Children)
                descendants.AddRange(child.GetDescendants());

            descendants = descendants.Distinct().ToList();
            return descendants;
        }
    }
}