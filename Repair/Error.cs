using Microsoft.Boogie;
using System.Collections.Generic;

namespace GPURepair.Repair
{
    public class Error
    {
        public ErrorType ErrorType { get; set; }

        public Implementation Implementation { get; set; }

        public IEnumerable<string> Variables { get; set; }
    }
}