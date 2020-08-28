namespace GPURepair.Instrumentation
{
    using System.Collections.Generic;
    using Microsoft.Boogie;

    public class AccessCollector : StandardVisitor
    {
        /// <summary>
        /// List of the global variables encountered so far.
        /// </summary>
        public List<GlobalVariable> Variables = new List<GlobalVariable>();

        /// <summary>
        /// This method is executed every time a variable is encountered in the parse tree.
        /// </summary>
        /// <param name="node">The variable.</param>
        /// <returns>The default return of the method.</returns>
        public override Variable VisitVariable(Variable node)
        {
            if (node is GlobalVariable)
                Variables.Add(node as GlobalVariable);
            return base.VisitVariable(node);
        }
    }
}
