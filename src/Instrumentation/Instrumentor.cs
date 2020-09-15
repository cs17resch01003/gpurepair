namespace GPURepair.Instrumentation
{
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Common;
    using GPURepair.Common.Diagnostics;
    using Microsoft.Basetypes;
    using Microsoft.Boogie;

    public class Instrumentor
    {
        /// <summary>
        /// The key used to mark instrumented assignments.
        /// </summary>
        private const string InstrumentationKey = "repair_instrumented";

        /// <summary>
        /// The key used to mark instrumented barriers.
        /// </summary>
        private const string BarrierKey = "repair_barrier";

        /// <summary>
        /// The source location key.
        /// </summary>
        private const string SourceLocationKey = "sourceloc_num";

        /// <summary>
        /// The block source key.
        /// </summary>
        private const string BlockSourceKey = "block_sourceloc";

        /// <summary>
        /// The program to be instrumented.
        /// </summary>
        private Microsoft.Boogie.Program program;

        /// <summary>
        /// The source language of the input program.
        /// </summary>
        private SourceLanguage sourceLanguage;

        /// <summary>
        /// Enables the grid-level barriers.
        /// </summary>
        private bool enableGridBarriers;

        /// <summary>
        /// The definition of the barrier.
        /// </summary>
        private Procedure barrier_definition;

        /// <summary>
        /// The definition of the grid-level barrier.
        /// </summary>
        private Procedure grid_barrier_definition;

        /// <summary>
        /// The total number of barriers created so far.
        /// </summary>
        private int barrier_count = 0;

        /// <summary>
        /// The total number of call commands encountered so far.
        /// </summary>
        private int call_commands = 0;

        /// <summary>
        /// Represents a bitvector of size 1.
        /// </summary>
        private LiteralExpr _1bv1;

        /// <summary>
        /// The graph analyzer.
        /// </summary>
        private GraphAnalyzer analyzer;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        /// <param name="sourceLanguage">The source language of the input program.</param>
        /// <param name="enableGridBarriers">Enables the grid-level barriers.</param>
        public Instrumentor(Microsoft.Boogie.Program program,
            SourceLanguage sourceLanguage, bool enableGridBarriers)
        {
            this.program = program;
            this.sourceLanguage = sourceLanguage;
            this.enableGridBarriers = enableGridBarriers;

            analyzer = new GraphAnalyzer(program);
            foreach (Declaration declaration in program.TopLevelDeclarations)
                if (declaration is Procedure)
                {
                    Procedure procedure = declaration as Procedure;
                    if (procedure.Name == "$bugle_barrier")
                        barrier_definition = procedure;
                    else if (procedure.Name == "$bugle_grid_barrier")
                        grid_barrier_definition = procedure;
                }

            _1bv1 = new LiteralExpr(Token.NoToken, BigNum.ONE, 1);

            Logger.Log($"Blocks;{program.Blocks().Count()}");
            Logger.Log($"Commands;{program.Blocks().Sum(x => x.Cmds.Count)}");
        }

        /// <summary>
        /// Instuments the program by adding the required barriers.
        /// </summary>
        public void Instrument()
        {
            bool program_modified;

            do program_modified = InstrumentCommand();
            while (program_modified);

            do program_modified = InstrumentBarrier();
            while (program_modified);

            InstrumentMergeNodes();

            Logger.Log($"CallCommands;{call_commands}");
            Logger.Log($"Barriers;{barrier_count}");
        }

        /// <summary>
        /// Adds one barrier at a time.
        /// </summary>
        /// <returns>True if the program has changed, False otherwise.</returns>
        private bool InstrumentCommand()
        {
            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    for (int i = 0; i < block.Cmds.Count; i++)
                    {
                        Cmd command = block.Cmds[i];

                        bool instrument = false;
                        int location = -1;

                        if (command is AssignCmd)
                        {
                            AssignCmd assign = command as AssignCmd;

                            // parse the expression to check for global variables
                            AccessCollector collector = new AccessCollector();
                            foreach (AssignLhs lhs in assign.Lhss)
                                collector.Visit(lhs);

                            foreach (Expr rhs in assign.Rhss)
                                collector.Visit(rhs);

                            instrument = collector.Variables.Any();
                        }
                        else if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;

                            // don't instrument invariants
                            if (call.callee.Contains("_invariant"))
                                continue;

                            // parse the expression to check for global variables
                            AccessCollector collector = new AccessCollector();
                            foreach (Expr _in in call.Ins)
                                collector.Visit(_in);

                            foreach (Expr _out in call.Outs)
                                collector.Visit(_out);

                            instrument = collector.Variables.Any();
                            location = QKeyValue.FindIntAttribute(call.Attributes, SourceLocationKey, -1);
                        }

                        // if there is a global variable
                        if (instrument)
                        {
                            // assert commands are used to store the metadata related to the actual command
                            AssertCmd assert = null;
                            if (i - 1 >= 0 && block.Cmds[i - 1] is AssertCmd)
                                assert = block.Cmds[i - 1] as AssertCmd;

                            if (assert == null)
                            {
                                LiteralExpr literal = new LiteralExpr(Token.NoToken, true);
                                assert = new AssertCmd(Token.NoToken, literal);

                                if (location != -1)
                                {
                                    LiteralExpr locationValue = new LiteralExpr(Token.NoToken, BigNum.FromInt(location));
                                    QKeyValue sourceLocationKey = new QKeyValue(Token.NoToken, SourceLocationKey, new List<object>() { locationValue }, null);

                                    assert.Attributes = sourceLocationKey;
                                }

                                block.Cmds.Insert(i, assert);
                                i = i + 1;
                            }

                            // and we haven't instrumented it already
                            if (!ContainsAttribute(assert, InstrumentationKey))
                            {
                                bool value = AddBarrier(implementation, block, i);
                                if (value == true)
                                {
                                    if (command is CallCmd)
                                        call_commands++;

                                    analyzer.LinkBarrier(implementation, block);
                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Instruments one uninstrumented barrier at a time.
        /// </summary>
        /// <returns>True if the program has changed, False otherwise.</returns>
        private bool InstrumentBarrier()
        {
            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    for (int i = 0; i < block.Cmds.Count; i++)
                    {
                        Cmd command = block.Cmds[i];
                        if (command is CallCmd)
                        {
                            // get the call comamnd
                            CallCmd call = command as CallCmd;
                            if (call.callee == "$bugle_barrier" || call.callee == "$bugle_grid_barrier")
                            {
                                if (!ContainsAttribute(call, BarrierKey))
                                {
                                    UpdateBarrier(implementation, block, i);
                                    analyzer.LinkBarrier(implementation, block);

                                    return true;
                                }
                            }
                        }
                    }
                }
            }

            return false;
        }

        /// <summary>
        /// Instrument the merge nodes in the program.
        /// </summary>
        private void InstrumentMergeNodes()
        {
            Dictionary<ProgramNode, MergeNodeType> nodes = analyzer.GetMergeNodes();
            foreach (ProgramNode node in nodes.Keys)
            {
                if (!node.Block.Cmds.Any())
                    continue;

                // skips asserts to preserve the invariants at the loop head
                int i = 1;
                while (node.Block.Cmds.Count > i && node.Block.Cmds[i] is AssertCmd)
                {
                    AssertCmd assert = node.Block.Cmds[i] as AssertCmd;
                    if (!ContainsAttribute(assert, "originated_from_invariant"))
                        break;

                    i = i + 1;
                }

                // the block should have source location information for instrumentation to work
                if (node.Block.Cmds[0] is AssertCmd)
                {
                    AssertCmd assert = node.Block.Cmds[0] as AssertCmd;
                    if (ContainsAttribute(assert, SourceLocationKey))
                    {
                        node.Block.Cmds.Insert(i, assert);
                        node.Block.Cmds.RemoveAt(0);

                        // insert a barrier at the beginning of the merge block
                        bool value = AddBarrier(node.Implementation, node.Block, i);
                        if (value == true)
                        {
                            analyzer.LinkBarrier(node.Implementation, node.Block);
                        }

                        if (nodes[node] == MergeNodeType.Loop)
                        {
                            // get the header nodes in the loop
                            List<Block> predecessors = new List<Block>();
                            foreach (Block block in program.Implementations.SelectMany(x => x.Blocks))
                                if (block.TransferCmd is GotoCmd)
                                {
                                    GotoCmd gotoCmd = block.TransferCmd as GotoCmd;
                                    if (gotoCmd.labelTargets.Contains(node.Block))
                                        predecessors.Add(block);
                                }

                            foreach (Block predecessor in predecessors)
                            {
                                if (analyzer.Graph.BackEdges.Any(x => x.Destination == node && x.Source.Block == predecessor))
                                    continue;

                                // identify the location of the initialize statement of the loop
                                int index = predecessor.Cmds.Count - 1;
                                while (index >= 0)
                                {
                                    if (predecessor.Cmds[index] is AssertCmd)
                                        break;
                                    index = index - 1;
                                }

                                if (index != -1)
                                {
                                    // create an assert from the block source location since initializers don't have source locations
                                    int location = QKeyValue.FindIntAttribute(assert.Attributes, SourceLocationKey, -1);
                                    if (location != -1)
                                    {
                                        assert = predecessor.Cmds[index] as AssertCmd;

                                        // becomes true when there is only one command in the block
                                        // in those cases, we don't need to need to add a new assert attribute
                                        if (!ContainsAttribute(assert, BlockSourceKey))
                                        {
                                            // create a new assert to store the location information
                                            LiteralExpr locationValue = new LiteralExpr(Token.NoToken, BigNum.FromInt(location));
                                            QKeyValue sourceLocationKey = new QKeyValue(Token.NoToken, SourceLocationKey, new List<object>() { locationValue }, null);

                                            LiteralExpr literal = new LiteralExpr(Token.NoToken, true);
                                            assert = new AssertCmd(Token.NoToken, literal);
                                            assert.Attributes = sourceLocationKey;

                                            index = index + 2;
                                            predecessor.Cmds.Insert(index, assert);
                                        }

                                        Implementation implementation = program.Implementations.First(x => x.Blocks.Contains(predecessor));

                                        // insert a barrier at the end of the header block
                                        value = AddBarrier(implementation, predecessor, index + 1);
                                        if (value == true)
                                        {
                                            analyzer.LinkBarrier(implementation, predecessor);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Checks if an assert command has the given attribute.
        /// </summary>
        /// <param name="assert">The assert command.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private bool ContainsAttribute(AssertCmd assert, string attribute)
        {
            QKeyValue keyValue = assert.Attributes;
            bool found = false;

            while (keyValue != null && !found)
            {
                if (attribute == keyValue.Key)
                    found = true;
                keyValue = keyValue.Next;
            }

            return found;
        }

        /// <summary>
        /// Checks if an call command has the given attribute.
        /// </summary>
        /// <param name="call">The call command.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private bool ContainsAttribute(CallCmd assert, string attribute)
        {
            QKeyValue keyValue = assert.Attributes;
            bool found = false;

            while (keyValue != null && !found)
            {
                if (attribute == keyValue.Key)
                    found = true;
                keyValue = keyValue.Next;
            }

            return found;
        }

        /// <summary>
        /// Adds the required barriers.
        /// </summary>
        /// <param name="implementation">The procedure where the blocks are being added.</param>
        /// <param name="block">The block containing the shared read/write command.</param>
        /// <param name="index">The index of the assign command.</param>
        private bool AddBarrier(Implementation implementation, Block block, int index)
        {
            // an assert statement is needed to obtain the source location
            // skipping instrumentation if it doesn't exist
            if (index - 1 < 0)
                return false;

            AssertCmd assert = block.Cmds[index - 1] as AssertCmd;
            if (assert == null)
                return false;

            // add the instrumentation key to the assert command
            QKeyValue assertAttribute = new QKeyValue(Token.NoToken, InstrumentationKey, new List<object>(), null);
            if (assert.Attributes != null)
                assert.Attributes.AddLast(assertAttribute);
            else
                assert.Attributes = assertAttribute;

            // get source location infromation
            int location = QKeyValue.FindIntAttribute(assert.Attributes, SourceLocationKey, -1);
            QKeyValue locationAttribute = null;

            if (location != -1)
            {
                LiteralExpr locationValue = new LiteralExpr(Token.NoToken, BigNum.FromInt(location));
                locationAttribute = new QKeyValue(Token.NoToken, SourceLocationKey, new List<object> { locationValue }, null);
            }

            // create a conditional barrier
            if (sourceLanguage == SourceLanguage.CUDA && enableGridBarriers)
            {
                Barrier barrier = CreateCallCommand(block, locationAttribute, index, "$bugle_barrier");
                Barrier grid_barrier = CreateCallCommand(block, locationAttribute, index, "$bugle_grid_barrier");

                InsertBarriers(implementation, block, index, barrier, grid_barrier);
            }
            else
            {
                Barrier barrier = CreateCallCommand(block, locationAttribute, index, "$bugle_barrier");
                InsertBarrier(implementation, block, index, barrier);
            }

            return true;
        }

        /// <summary>
        /// Creates a call command for the given procedure name.
        /// </summary>
        /// <param name="block">The block containing the shared read/write command.</param>
        /// <param name="locationAttribute">The location attribute.</param>
        /// <param name="index">The index of the assign command.</param>
        /// <param name="procedureName">The procedure name.</param>
        /// <returns>The barrier name and the call command.</returns>
        private Barrier CreateCallCommand(Block block, QKeyValue locationAttribute, int index, string procedureName)
        {
            if (procedureName == "$bugle_barrier")
                CreateBugleBarrier();
            else if (procedureName == "$bugle_grid_barrier")
                CreateGridLevelBugleBarrier();

            // create the barrier attributes
            string barrierName = "b" + (++barrier_count);
            QKeyValue barrierAttribute = new QKeyValue(Token.NoToken, BarrierKey, new List<object>() { barrierName }, null);

            // create the call barrier command if it doesn't exist
            Barrier barrier = new Barrier(barrierName);
            if (index - 2 >= 0 && block.Cmds[index - 2] is CallCmd && (block.Cmds[index - 2] as CallCmd).callee == procedureName)
            {
                barrier.Call = block.Cmds[index - 2] as CallCmd;
                barrier.Call.Attributes.AddLast(barrierAttribute);
                barrier.Existing = true;
            }
            else
            {
                QKeyValue instrumentedAttribute = new QKeyValue(Token.NoToken, InstrumentationKey, new List<object>(), null);
                List<Expr> arguments = procedureName == "$bugle_barrier" ? new List<Expr>() { _1bv1, _1bv1 } : new List<Expr>();

                barrier.Call = new CallCmd(Token.NoToken, procedureName, arguments, new List<IdentifierExpr>(), barrierAttribute);
                barrier.Call.Attributes.AddLast(instrumentedAttribute);
                if (locationAttribute != null)
                    barrier.Call.Attributes.AddLast(locationAttribute);
            }

            return barrier;
        }

        /// <summary>
        /// Inserts the barrier at the specified index.
        /// </summary>
        /// <param name="implementation">The procedure where the blocks are being added.</param>
        /// <param name="block">The block containing the shared read/write command.</param>
        /// <param name="index">The index of the assign command.</param>
        /// <param name="barrier">The barrier to insert.</param>
        private Block InsertBarrier(Implementation implementation, Block block, int index, Barrier barrier)
        {
            if (barrier.Existing == false)
            {
                block.Cmds.Insert(index - 1, barrier.Call);
                index = index + 1;
            }

            GlobalVariable variable = CreateGlobalVariable(barrier.Name);
            QKeyValue partition = new QKeyValue(Token.NoToken, "partition", new List<object>(), null);

            // move the commands to a new block to create a conditional barrier
            Block end = new Block(Token.NoToken, barrier.Name + "_end", new List<Cmd>(), block.TransferCmd);
            end.Cmds.AddRange(block.Cmds.GetRange(index - 1, block.Cmds.Count - index + 1));

            // create the true block
            Block true_block = new Block(Token.NoToken, barrier.Name + "_true", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            true_block.Cmds.Add(new AssumeCmd(Token.NoToken, new IdentifierExpr(Token.NoToken, variable), partition));
            true_block.Cmds.Add(barrier.Call);

            // create the false block
            Block false_block = new Block(Token.NoToken, barrier.Name + "_false", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            false_block.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.Not(new IdentifierExpr(Token.NoToken, variable)), partition));

            // add the conditional barrier
            block.TransferCmd = new GotoCmd(Token.NoToken, new List<Block>() { true_block, false_block });
            block.Cmds.RemoveRange(index - 2, block.Cmds.Count - index + 2);

            implementation.Blocks.Add(true_block);
            implementation.Blocks.Add(false_block);
            implementation.Blocks.Add(end);

            return end;
        }

        /// <summary>
        /// Inserts the barriers at the specified index.
        /// </summary>
        /// <param name="implementation">The procedure where the blocks are being added.</param>
        /// <param name="block">The block containing the shared read/write command.</param>
        /// <param name="index">The index of the assign command.</param>
        /// <param name="barrier">The barrier to insert.</param>
        /// <param name="grid_barrier">The grid-level barrier to insert.</param>
        private void InsertBarriers(Implementation implementation, Block block, int index, Barrier barrier, Barrier grid_barrier)
        {
            // we always instrument the thread-level barrier before the grid-level barrier
            barrier.Existing = barrier.Existing || grid_barrier.Existing;
            grid_barrier.Existing = false;

            Block end = InsertBarrier(implementation, block, index, barrier);
            InsertBarrier(implementation, end, 1, grid_barrier); // the command containing the shared read/write will be at position 1
        }

        /// <summary>
        /// Updates the barrier to make it optional.
        /// </summary>
        /// <param name="implementation">The procedure where the blocks are being added.</param>
        /// <param name="block">The block contains the shared write.</param>
        /// <param name="index">The index of the assign command.</param>
        private void UpdateBarrier(Implementation implementation, Block block, int index)
        {
            string barrierName = "b" + (++barrier_count);
            CallCmd call = block.Cmds[index] as CallCmd;

            // add the instrumentation key
            QKeyValue barrierAttribute = new QKeyValue(Token.NoToken, BarrierKey, new List<object>() { barrierName }, null);
            call.Attributes.AddLast(barrierAttribute);

            GlobalVariable variable = CreateGlobalVariable(barrierName);
            QKeyValue partition = new QKeyValue(Token.NoToken, "partition", new List<object>(), null);

            // move the commands to a new block to create a conditional barrier
            Block end = new Block(Token.NoToken, barrierName + "_end", new List<Cmd>(), block.TransferCmd);
            end.Cmds.AddRange(block.Cmds.GetRange(index + 1, block.Cmds.Count - index - 1));

            // create the true block
            Block true_block = new Block(Token.NoToken, barrierName + "_true", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            true_block.Cmds.Add(new AssumeCmd(Token.NoToken, new IdentifierExpr(Token.NoToken, variable), partition));
            true_block.Cmds.Add(call);

            // create the false block
            Block false_block = new Block(Token.NoToken, barrierName + "_false", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            false_block.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.Not(new IdentifierExpr(Token.NoToken, variable)), partition));

            // add the conditional barrier
            block.TransferCmd = new GotoCmd(Token.NoToken, new List<Block>() { true_block, false_block });
            block.Cmds.RemoveRange(index, block.Cmds.Count - index);

            implementation.Blocks.Add(true_block);
            implementation.Blocks.Add(false_block);
            implementation.Blocks.Add(end);
        }

        /// <summary>
        /// Creates the definition for the barrier if it doesn't exist.
        /// </summary>
        private void CreateBugleBarrier()
        {
            if (barrier_definition != null)
                return;

            List<Variable> variables = new List<Variable>();
            variables.Add(new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "$0", Type.GetBvType(1)), false));
            variables.Add(new Formal(Token.NoToken, new TypedIdent(Token.NoToken, "$1", Type.GetBvType(1)), false));

            Procedure procedure = new Procedure(Token.NoToken, "$bugle_barrier",
                new List<TypeVariable>(), variables, new List<Variable>(), new List<Requires>(),
                new List<IdentifierExpr>(), new List<Ensures>());

            Dictionary<string, object> attributes = new Dictionary<string, object>();
            attributes.Add("barrier", null);

            procedure.Attributes = ConvertAttributes(attributes);
            program.AddTopLevelDeclaration(procedure);

            barrier_definition = procedure;
        }

        /// <summary>
        /// Creates the definition for the grid-level barrier if it doesn't exist.
        /// </summary>
        private void CreateGridLevelBugleBarrier()
        {
            if (grid_barrier_definition != null)
                return;

            List<Variable> variables = new List<Variable>();
            Procedure procedure = new Procedure(Token.NoToken, "$bugle_grid_barrier",
                new List<TypeVariable>(), variables, new List<Variable>(), new List<Requires>(),
                new List<IdentifierExpr>(), new List<Ensures>());

            Dictionary<string, object> attributes = new Dictionary<string, object>();
            attributes.Add("grid_barrier", null);

            procedure.Attributes = ConvertAttributes(attributes);
            program.AddTopLevelDeclaration(procedure);

            grid_barrier_definition = procedure;
        }

        /// <summary>
        /// Creates a global variable with the given name.
        /// </summary>
        /// <param name="name">The name of the variable.</param>
        /// <returns>A reference to the variable.</returns>
        private GlobalVariable CreateGlobalVariable(string name)
        {
            GlobalVariable variable = new GlobalVariable(Token.NoToken,
                new TypedIdent(Token.NoToken, name, Type.Bool));

            Dictionary<string, object> attributes = new Dictionary<string, object>();
            attributes.Add("repair", null);
            attributes.Add("race_checking", null);
            attributes.Add("global", null);
            attributes.Add("elem_width", Expr.Literal(32));
            attributes.Add("source_elem_width", Expr.Literal(32));
            attributes.Add("source_dimensions", "*");

            variable.Attributes = ConvertAttributes(attributes);
            program.AddTopLevelDeclaration(variable);

            return variable;
        }

        /// <summary>
        /// Generates a key-value chain from the given attributes.
        /// </summary>
        /// <param name="attributes">The attributes.</param>
        /// <returns>The first node in the attribute chain.</returns>
        private QKeyValue ConvertAttributes(Dictionary<string, object> attributes)
        {
            QKeyValue value = null;
            foreach (KeyValuePair<string, object> pair in attributes)
            {
                List<object> array = new List<object>();
                if (pair.Value != null)
                    array.Add(pair.Value);

                QKeyValue temp = new QKeyValue(Token.NoToken, pair.Key, array, value);
                value = temp;
            }

            return value;
        }

        /// <summary>
        /// Represents a barrier.
        /// </summary>
        private class Barrier
        {
            /// <summary>
            /// The constructor.
            /// </summary>
            /// <param name="name">The name of the barrier.</param>
            public Barrier(string name)
            {
                Name = name;
            }

            /// <summary>
            /// The name of the barrier.
            /// </summary>
            public string Name { get; set; }

            /// <summary>
            /// The call command associated with the barrier.
            /// </summary>
            public CallCmd Call { get; set; }

            /// <summary>
            /// Indicates if an existing barrier was used.
            /// </summary>
            public bool Existing { get; set; }
        }
    }
}
