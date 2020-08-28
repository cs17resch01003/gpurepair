namespace GPURepair.Repair
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using GPURepair.Common.Diagnostics;
    using GPURepair.Repair.Metadata;

    public class SummaryGenerator
    {
        /// <summary>
        /// Generate a summary for the barrier changes.
        /// </summary>
        /// <param name="assignments">The assignments.</param>
        /// <param name="filename">The file path.</param>
        /// <return>The changes.</return>
        public IEnumerable<string> GenerateSummary(Dictionary<string, bool> assignments, string filename)
        {
            List<string> lines = new List<string>();
            List<string> changes = new List<string>();

            int barriersBetweenCalls = 0;
            foreach (string barrierName in assignments.Keys)
            {
                Barrier barrier = ProgramMetadata.Barriers[barrierName];
                List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];

                string location = ToString(locations);
                if (!barrier.Generated && !assignments[barrierName])
                {
                    lines.Add(string.Format("Remove the barrier at location {0}.", location));
                    changes.Add(location);

                    if (locations.Count > 1)
                        barriersBetweenCalls++;
                }
                else if (barrier.Generated && assignments[barrierName])
                {
                    lines.Add(string.Format("Add a barrier at location {0}.", location));
                    changes.Add(location);

                    if (locations.Count > 1)
                        barriersBetweenCalls++;
                }
            }

            Logger.Log($"BarriersBetweenCalls;{barriersBetweenCalls}");
            if (lines.Count != 0)
                File.AppendAllLines(filename, lines);

            return changes;
        }

        /// <summary>
        /// Represents the location in a string format.
        /// </summary>
        /// <param name="locations"> The location stack.</param>
        /// <returns>A string representation of the location.</returns>
        private static string ToString(List<Location> locations)
        {
            return string.Join(" => ", locations.Select(x => x.ToString()));
        }
    }
}
