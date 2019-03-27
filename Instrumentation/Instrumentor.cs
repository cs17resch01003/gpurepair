using Microsoft.Basetypes;
using Microsoft.Boogie;
using System.Collections.Generic;
using System.Linq;

namespace GPURepair.Instrumentation
{
    public class Instrumentor
    {
        /// <summary>
        /// The key used to mark instrumented assignments.
        /// </summary>
        private const string InstrumentationKey = "repair_instrumented";

        /// <summary>
        /// The program to be instrumented.
        /// </summary>
        private Microsoft.Boogie.Program program;

        /// <summary>
        /// The definition of the barrier.
        /// </summary>
        private Procedure barrier_definition;

        /// <summary>
        /// The total number of barriers created so far.
        /// </summary>
        private int barrier_count = 0;

        /// <summary>
        /// Represents a bitvector of size 1.
        /// </summary>
        private LiteralExpr _1bv1;

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public Instrumentor(Microsoft.Boogie.Program program)
        {
            this.program = program;

            foreach (Declaration declaration in program.TopLevelDeclarations)
                if (declaration is Procedure)
                {
                    Procedure procedure = declaration as Procedure;
                    if (procedure.Name == "$bugle_barrier")
                        barrier_definition = procedure;
                }

            _1bv1 = new LiteralExpr(Token.NoToken, BigNum.ONE, 1);
        }

        /// <summary>
        /// Instuments the program by adding the required barriers.
        /// </summary>
        public void Instrument()
        {
            bool program_modified = false;
            
            do program_modified = InstrumentNext();
            while (program_modified);
        }

        /// <summary>
        /// Adds one barrier at a time.
        /// </summary>
        /// <returns>True if the program has changed, False otherwise.</returns>
        private bool InstrumentNext()
        {
            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    for (int i = 0; i < block.Cmds.Count; i++)
                    {
                        Cmd command = block.Cmds[i];
                        if (command is AssignCmd)
                        {
                            // get the assign and assert commands
                            // assert commands are used to store the metadata related to the assign command
                            AssignCmd assign = command as AssignCmd;
                            AssertCmd assert = null;
                            if (block.Cmds[i - 1] is AssertCmd)
                                assert = block.Cmds[i - 1] as AssertCmd;

                            // parse the expression to check for global variables
                            AccessCollector collector = new AccessCollector();
                            foreach (AssignLhs lhs in assign.Lhss)
                                collector.Visit(lhs);

                            // if there is a global variable
                            if (collector.Variables.Any())
                            {
                                // and we haven't instrumented it already
                                if (assert == null || !ContainsAttribute(assert, InstrumentationKey))
                                {
                                    AddBarrier(implementation, block, i);
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
        /// Adds the required barrier.
        /// </summary>
        /// <param name="implementation">The procedure where the blocks are being added.</param>
        /// <param name="block">The block contains the shared write.</param>
        /// <param name="index">The index of the assign command.</param>
        private void AddBarrier(Implementation implementation, Block block, int index)
        {
            CreateBugleBarrier();
            string barrier_name = "b" + (++barrier_count);

            // get the assign and assert commands
            // create the assert command if it doesn't exist
            AssignCmd assign = block.Cmds[index] as AssignCmd;
            AssertCmd assert;

            if (block.Cmds[index - 1] is AssertCmd)
                assert = block.Cmds[index - 1] as AssertCmd;
            else
            {
                assert = new AssertCmd(Token.NoToken, Expr.Literal(true));
                block.Cmds.Insert(index++, assert);
            }

            // add the instrumentation attribute to the assert command
            assert.Attributes.AddLast(new QKeyValue(Token.NoToken, InstrumentationKey, new List<object>(), null));
            GlobalVariable variable = CreateGlobalVariable(barrier_name);

            QKeyValue partition = new QKeyValue(Token.NoToken, "partition", new List<object>(), null);

            // move the commands to a new block to create a conditional barrier
            Block end = new Block(Token.NoToken, barrier_name + "_end", new List<Cmd>(), block.TransferCmd);
            end.Cmds.AddRange(block.Cmds.GetRange(index - 1, block.Cmds.Count - index + 1));

            // create the true block
            Block true_block = new Block(Token.NoToken, barrier_name + "_true", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            true_block.Cmds.Add(new AssumeCmd(Token.NoToken, new IdentifierExpr(Token.NoToken, variable), partition));
            true_block.Cmds.Add(new CallCmd(Token.NoToken, "$bugle_barrier", new List<Expr>() { _1bv1, _1bv1 }, new List<IdentifierExpr>()));

            // create the false block
            Block false_block = new Block(Token.NoToken, barrier_name + "_false", new List<Cmd>(), new GotoCmd(Token.NoToken, new List<Block>() { end }));
            false_block.Cmds.Add(new AssumeCmd(Token.NoToken, Expr.Not(new IdentifierExpr(Token.NoToken, variable)), partition));

            // add the conditional barrier
            block.TransferCmd = new GotoCmd(Token.NoToken, new List<Block>() { true_block, false_block });
            block.Cmds.RemoveRange(index - 1, block.Cmds.Count - index + 1);

            implementation.Blocks.Add(true_block);
            implementation.Blocks.Add(false_block);
            implementation.Blocks.Add(end);
        }

        /// <summary>
        /// Creates the definition for a barrier if it doesn't exist.
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
    }
}