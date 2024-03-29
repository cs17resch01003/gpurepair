﻿namespace GPURepair.Repair.Metadata
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text.RegularExpressions;
    using GPURepair.Common;
    using GPURepair.Common.Diagnostics;
    using GPURepair.Repair.Diagnostics;
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
        public static Dictionary<int, LocationChain> Locations { get; private set; }

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
        /// <param name="useAxioms">Use axioms for instrumentation instead of variable assignments.</param>
        public static void PopulateMetadata(string filePath, bool useAxioms)
        {
            Program program = BoogieUtilities.ReadFile(filePath);

            PopulateLocations(filePath.Replace(".cbpl", ".loc"));
            PopulateBarriers(program, useAxioms);
            PopulateLoopInformation(program);

            ClauseLogger.LogWeights();
            Logger.Log($"GridLevelBarriers;{Barriers.Values.Where(x => x.GridLevel).Count()}");
            Logger.Log($"LoopBarriers;{Barriers.Values.Where(x => x.LoopDepth > 0).Count()}");
        }

        /// <summary>
        /// Populates the locations.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        private static void PopulateLocations(string filePath)
        {
            Locations = new Dictionary<int, LocationChain>();

            using (StreamReader sr = new StreamReader(filePath))
            {
                string[] locations = sr.ReadLine().Split(new char[] { '\x1D' });
                for (int i = 0; i < locations.Length; i++)
                {
                    string line = locations[i];
                    string[] chain = line.Split(new char[] { '\x1E' });

                    LocationChain locationChain = new LocationChain();
                    locationChain.SourceLocation = i;

                    foreach (string c in chain)
                    {
                        if (c != string.Empty)
                        {
                            string[] source = c.Split(new char[] { '\x1F' });
                            locationChain.Add(new Location
                            {
                                Line = Convert.ToInt32(source[0]),
                                Column = Convert.ToInt32(source[1]),
                                File = source[2],
                                Directory = new DirectoryInfo(source[3]).FullName,
                            });
                        }
                    }

                    if (locationChain.Any())
                        Locations.Add(i, locationChain);
                }
            }
        }

        /// <summary>
        /// Populates the barriers.
        /// </summary>
        /// <param name="program">The given program.</param>
        /// <param name="useAxioms">Use axioms for instrumentation instead of variable assignments.</param>
        private static void PopulateBarriers(Program program, bool useAxioms)
        {
            Barriers = new Dictionary<string, Barrier>();
            Regex regex = new Regex(@"^b\d+$");

            if (useAxioms)
            {
                program.Constants.Where(x => regex.IsMatch(x.Name))
                    .ToList().ForEach(x => Barriers.Add(x.Name, new Barrier(x.Name)));
            }
            else
            {
                program.TopLevelDeclarations.Where(x => x is GlobalVariable)
                    .Select(x => x as GlobalVariable).Where(x => regex.IsMatch(x.Name))
                    .ToList().ForEach(x => Barriers.Add(x.Name, new Barrier(x.Name)));
            }

            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    foreach (Cmd command in block.Cmds)
                    {
                        if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;
                            if (call.callee.Contains("$bugle_barrier") || call.callee.Contains("$bugle_grid_barrier"))
                            {
                                int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                                string barrierName = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                                bool generated = ContainsAttribute(call, "repair_instrumented");

                                if (barrierName != null)
                                {
                                    Barrier barrier = Barriers[barrierName];

                                    barrier.Implementation = implementation;
                                    barrier.Block = block;
                                    barrier.Call = call;
                                    barrier.SourceLocation = location;
                                    barrier.Generated = generated;
                                    barrier.GridLevel = call.callee.Contains("$bugle_grid_barrier");
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Populates the loop information.
        /// </summary>
        /// <param name="program">The given program.</param>
        private static void PopulateLoopInformation(Program program)
        {
            ProgramGraph graph = new ProgramGraph(program);

            BackEdges = graph.BackEdges;
            LoopBarriers = new Dictionary<BackEdge, List<Barrier>>();

            foreach (BackEdge edge in graph.BackEdges)
            {
                List<ProgramNode> loopNodes = graph.GetLoopNodes(edge);
                List<Block> loopBlocks = loopNodes.Select(x => x.Block).ToList();

                IEnumerable<Barrier> barriers = Barriers.Values.Where(x => loopBlocks.Contains(x.Block));
                LoopBarriers.Add(edge, barriers.ToList());
            }

            foreach (Barrier barrier in Barriers.Values)
                barrier.LoopDepth = LoopBarriers.Where(x => x.Value.Contains(barrier)).Count();

            // consider barriers instrumented in the header node as a part of the loop
            // these barriers will not have a loop depth since they are outside the loop
            foreach (BackEdge edge in graph.BackEdges)
            {
                foreach (ProgramNode predecessor in edge.Destination.Predecessors)
                {
                    // ignore the other end of the back-edge
                    if (predecessor == edge.Source)
                        continue;

                    // identify the last thread-level and grid-level barrier in the predecessor
                    int? thread = null, grid = null;

                    int location = predecessor.Block.Cmds.Count - 1;
                    while (location >= 0)
                    {
                        if (predecessor.Block.Cmds[location] is CallCmd)
                        {
                            CallCmd command = predecessor.Block.Cmds[location] as CallCmd;
                            if (thread == null && command.callee.Contains("$bugle_barrier"))
                                thread = location;

                            if (grid == null && command.callee.Contains("bugle_grid_barrier"))
                                grid = location;
                        }

                        if (thread != null && grid != null)
                            break;
                        location = location - 1;
                    }

                    if (thread != null)
                    {
                        CallCmd command = predecessor.Block.Cmds[thread.Value] as CallCmd;
                        Barrier barrier = Barriers.Where(x => x.Value.Call == command).Select(x => x.Value).FirstOrDefault();
                        if (barrier != null && !LoopBarriers[edge].Contains(barrier))
                            LoopBarriers[edge].Add(barrier);
                    }

                    if (grid != null)
                    {
                        CallCmd command = predecessor.Block.Cmds[grid.Value] as CallCmd;
                        Barrier barrier = Barriers.Where(x => x.Value.Call == command).Select(x => x.Value).FirstOrDefault();
                        if (barrier != null && !LoopBarriers[edge].Contains(barrier))
                            LoopBarriers[edge].Add(barrier);
                    }
                }
            }
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
