namespace GPURepair.Repair.Metadata
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;
    using GPURepair.Common;
    using GPURepair.Common.Diagnostics;
    using Microsoft.Boogie;

    public static class ProgramMetadata
    {
        /// <summary>
        /// The barriers in the program.
        /// </summary>
        public static Dictionary<string, Barrier> Barriers { get; private set; }

        /// <summary>
        /// The source locations in the program.
        /// </summary>
        public static Dictionary<int, List<Location>> Locations { get; private set; }

        /// <summary>
        /// The back edges in the program.
        /// </summary>
        public static List<BackEdge> BackEdges { get; private set; }

        /// <summary>
        /// Maps the loops with the barriers inside the loops.
        /// </summary>
        public static Dictionary<BackEdge, List<Barrier>> LoopBarriers { get; private set; }

        /// <summary>
        /// Returns true if grid-level barriers exists in the program.
        /// </summary>
        public static bool GridLevelBarriersExists
        {
            get
            {
                return Barriers.Values.Any(x => x.GridLevel);
            }
        }

        /// <summary>
        /// Populates the metadata.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public static void PopulateMetadata(string filePath)
        {
            Microsoft.Boogie.Program program = BoogieUtilities.ReadFile(filePath);

            PopulateLocations(filePath.Replace(".cbpl", ".loc"));
            PopulateBarriers(program);
            PopulateLoopInformation(program);

            Logger.Log($"GridLevelBarriers;{Barriers.Values.Where(x => x.GridLevel).Count()}");
            Logger.Log($"BarriersInsideLoop;{Barriers.Values.Where(x => x.LoopDepth > 0).Count()}");
        }

        /// <summary>
        /// Gets the location based on the file, line and column.
        /// </summary>
        /// <param name="directory">The directory.</param>
        /// <param name="file">The file.</param>
        /// <param name="line">The line.</param>
        /// <param name="column">The column.</param>
        /// <returns>The location.</returns>
        public static Location GetLocation(string directory, string file, int line, int column)
        {
            foreach (List<Location> locations in Locations.Values)
                foreach (Location location in locations)
                    if (location.Directory == directory && location.File == file &&
                            location.Line == line && location.Column == column)
                        return location;

            return null;
        }

        /// <summary>
        /// Populates the locations.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        private static void PopulateLocations(string filePath)
        {
            Locations = new Dictionary<int, List<Location>>();

            using (StreamReader sr = new StreamReader(filePath))
            {
                string[] locations = sr.ReadLine().Split(new char[] { '\x1D' });
                for (int i = 0; i < locations.Length; i++)
                {
                    string line = locations[i];
                    string[] chain = line.Split(new char[] { '\x1E' });

                    Locations.Add(i, new List<Location>());
                    foreach (string c in chain)
                    {
                        if (c != string.Empty)
                        {
                            string[] source = c.Split(new char[] { '\x1F' });
                            Locations[i].Add(new Location
                            {
                                SourceLocation = i,
                                Line = Convert.ToInt32(source[0]),
                                Column = Convert.ToInt32(source[1]),
                                File = source[2],
                                Directory = new DirectoryInfo(source[3]).FullName,
                            });
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Populates the barriers.
        /// </summary>
        /// <param name="program">The given program.</param>
        private static void PopulateBarriers(Microsoft.Boogie.Program program)
        {
            Barriers = new Dictionary<string, Barrier>();
            Regex regex = new Regex(@"^b\d+$");

            foreach (Declaration declaration in program.TopLevelDeclarations)
                if (declaration is GlobalVariable)
                {
                    GlobalVariable globalVariable = declaration as GlobalVariable;
                    if (regex.IsMatch(globalVariable.Name))
                    {
                        Barriers.Add(globalVariable.Name, new Barrier()
                        {
                            Name = globalVariable.Name,
                            Variable = globalVariable
                        });
                    }
                }

            foreach (Implementation implementation in program.Implementations)
                foreach (Block block in implementation.Blocks)
                    foreach (Cmd command in block.Cmds)
                        if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;
                            if (call.callee.Contains("$bugle_barrier") || call.callee.Contains("$bugle_grid_barrier"))
                            {
                                int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                                string barrierName = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                                bool generated = ContainsAttribute(call, "repair_instrumented");

                                Barrier barrier = Barriers[barrierName];

                                barrier.Implementation = implementation;
                                barrier.Block = block;
                                barrier.SourceLocation = location;
                                barrier.Generated = generated;
                                barrier.GridLevel = call.callee.Contains("$bugle_grid_barrier");
                            }
                        }
        }

        /// <summary>
        /// Populates the loop information.
        /// </summary>
        /// <param name="program">The given program.</param>
        private static void PopulateLoopInformation(Microsoft.Boogie.Program program)
        {
            ProgramGraph graph = new ProgramGraph(program);

            BackEdges = graph.BackEdges;
            LoopBarriers = new Dictionary<BackEdge, List<Barrier>>();

            foreach (BackEdge edge in graph.BackEdges)
            {
                List<ProgramNode> loopNodes = graph.GetLoopNodes(edge);
                List<Block> loopBlocks = loopNodes.Select(x => x.Block).ToList();

                List<Barrier> barriers = Barriers.Values
                    .Where(x => loopBlocks.Contains(x.Block)).ToList();
                LoopBarriers.Add(edge, barriers);
            }

            foreach (Barrier barrier in Barriers.Values)
                barrier.LoopDepth = LoopBarriers.Where(x => x.Value.Contains(barrier)).Count();
        }

        /// <summary>
        /// Checks if an call command has the given attribute.
        /// </summary>
        /// <param name="call">The call command.</param>
        /// <param name="attribute">The attribute.</param>
        /// <returns>True if the attribute exists, False otherwise.</returns>
        private static bool ContainsAttribute(CallCmd assert, string attribute)
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
    }
}
