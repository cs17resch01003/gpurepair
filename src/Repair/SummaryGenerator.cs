using System.Collections.Generic;
using System.IO;
using System.Linq;
using GPURepair.Repair.Metadata;

namespace GPURepair.Repair
{
    public class SummaryGenerator
    {
        /// <summary>
        /// Generate a summary for the barrier changes.
        /// </summary>
        /// <param name="assignments">The assignments.</param>
        /// <param name="filename">The file path.</param>
        /// <return>The changes.</return>
        public IEnumerable<Location> GenerateSummary(Dictionary<string, bool> assignments, string filename)
        {
            List<string> lines = new List<string>();
            List<Location> changes = new List<Location>();

            int barriersBetweenCalls = 0;
            foreach (string barrierName in assignments.Keys)
            {
                Barrier barrier = ProgramMetadata.Barriers[barrierName];
                List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];

                Location location = locations.Last();
                if (!barrier.Generated && !assignments[barrierName])
                {
                    lines.Add(string.Format("Remove the barrier at location {0}.", location.ToString()));
                    changes.Add(location);

                    if (locations.Count > 1)
                        barriersBetweenCalls++;
                }
                else if (barrier.Generated && assignments[barrierName])
                {
                    lines.Add(string.Format("Add a barrier at location {0}.", location.ToString()));
                    changes.Add(location);

                    if (locations.Count > 1)
                        barriersBetweenCalls++;
                }
            }

            if (lines.Count != 0)
                File.AppendAllLines(filename, lines);

            return changes;
        }
    }
}
