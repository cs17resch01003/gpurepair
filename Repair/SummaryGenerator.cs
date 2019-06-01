using Microsoft.Boogie;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace GPURepair.Repair
{
    public class SummaryGenerator
    {
        /// <summary>
        /// The repaired program.
        /// </summary>
        private Microsoft.Boogie.Program program;

        /// <summary>
        /// The barrier metadata.
        /// </summary>
        private List<BarrierMetadata> barriers = new List<BarrierMetadata>();

        /// <summary>
        /// Initialize the variables.
        /// </summary>
        /// <param name="program">The source Boogie program.</param>
        public SummaryGenerator(Microsoft.Boogie.Program program, Dictionary<string, bool> assignments)
        {
            this.program = program;
            foreach (string key in assignments.Keys)
            {
                barriers.Add(new BarrierMetadata()
                {
                    BarrierName = key,
                    Assignment = assignments[key]
                });
            }
        }

        /// <summary>
        /// Generate a summary for the barrier changes.
        /// </summary>
        /// <param name="filename">The file path.</param>
        /// <return>The changes.</return>
        public IEnumerable<Location> GenerateSummary(string filename)
        {
            PopulateMetadata();

            List<string> lines = new List<string>();
            List<Location> changes = new List<Location>();

            foreach (BarrierMetadata metadata in barriers)
            {
                Location location = GetLocation(metadata.SourceLocation, filename.Replace(".summary", ".loc"));

                if (!metadata.Generated && !metadata.Assignment)
                {
                    lines.Add(string.Format("Remove the barrier at location {0}.", location.ToString()));
                    changes.Add(location);
                }
                else if (metadata.Generated && metadata.Assignment)
                {
                    lines.Add(string.Format("Add a barrier at location {0}.", location.ToString()));
                    changes.Add(location);
                }
            }

            if (lines.Count != 0)
                File.AppendAllLines(filename, lines);

            return changes;
        }

        /// <summary>
        /// Populate the barrier metadata.
        /// </summary>
        private void PopulateMetadata()
        {
            foreach (Implementation implementation in program.Implementations)
            {
                foreach (Block block in implementation.Blocks)
                {
                    foreach (Cmd command in block.Cmds)
                    {
                        if (command is CallCmd)
                        {
                            CallCmd call = command as CallCmd;
                            if (call.callee.Contains("$bugle_barrier"))
                            {
                                int location = QKeyValue.FindIntAttribute(call.Attributes, "sourceloc_num", -1);
                                string barrier = QKeyValue.FindStringAttribute(call.Attributes, "repair_barrier");
                                bool generated = ContainsAttribute(call, "repair_instrumented");

                                BarrierMetadata metadata = barriers.FirstOrDefault(x => x.BarrierName == barrier);

                                // by default unnecessary barriers are removed
                                if (metadata == null)
                                {
                                    metadata = new BarrierMetadata()
                                    {
                                        BarrierName = barrier,
                                        Assignment = false
                                    };

                                    barriers.Add(metadata);
                                }

                                metadata.SourceLocation = location;
                                metadata.Generated = generated;
                            }
                        }
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
        /// Gets the location from the .loc file.
        /// </summary>
        /// <param name="sourceLocation">The source location line.</param>
        /// <param name="filename">The filename.</param>
        /// <returns>The location.</returns>
        private Location GetLocation(int sourceLocation, string filename)
        {
            using (StreamReader sr = new StreamReader(filename))
            {
                string line = sr.ReadLine().Split(new char[] { '\x1D' })[sourceLocation];
                string[] chain = line.Split(new char[] { '\x1E' });

                foreach (string c in chain)
                {
                    if (c != string.Empty)
                    {
                        string[] source = c.Split(new char[] { '\x1F' });
                        return new Location()
                        {
                            Line = Convert.ToInt32(source[0]),
                            Column = Convert.ToInt32(source[1]),
                            File = source[2],
                            Directory = source[3]
                        };
                    }
                }
            }

            return null;
        }

        /// <summary>
        /// The metadata related to a barrier.
        /// </summary>
        private class BarrierMetadata
        {
            /// <summary>
            /// The name of the barrier.
            /// </summary>
            public string BarrierName { get; set; }

            /// <summary>
            /// The barrier assignment.
            /// </summary>
            public bool Assignment { get; set; }

            /// <summary>
            /// Source location of the barrier.
            /// </summary>
            public int SourceLocation { get; set; }

            /// <summary>
            /// Indicates if the barrier was generated.
            /// </summary>
            public bool Generated { get; set; }
        }

        /// <summary>
        /// The location.
        /// </summary>
        public class Location
        {
            /// <summary>
            /// The line in the source file.
            /// </summary>
            public int Line { get; set; }

            /// <summary>
            /// The column in the source file.
            /// </summary>
            public int Column { get; set; }

            /// <summary>
            /// The source file.
            /// </summary>
            public string File { get; set; }

            /// <summary>
            /// The source directory.
            /// </summary>
            public string Directory { get; set; }

            /// <summary>
            /// Converts the object to a string.
            /// </summary>
            /// <returns>The string representation.</returns>
            public override string ToString()
            {
                return string.Format("{0} {1}", File, Line);
            }
        }
    }
}