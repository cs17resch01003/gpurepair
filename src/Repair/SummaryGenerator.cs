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
            int repaired_weight = 0;
            foreach (string barrierName in assignments.Where(x => x.Value == true).Select(x => x.Key))
            {
                Barrier barrier = ProgramMetadata.Barriers[barrierName];
                repaired_weight += barrier.Weight;
            }

            int source_weight = 0;
            foreach (Barrier barrier in ProgramMetadata.Barriers.Values)
            {
                if (!barrier.Generated)
                    source_weight += barrier.Weight;
            }

            Logger.Log($"SourceWeight;{source_weight}");
            Logger.Log($"RepairedWeight;{repaired_weight}");

            List<string> lines = new List<string>();
            List<string> changes = new List<string>();

            int solutionCalls = 0;
            int solutionGrid = 0;
            int solutionLoop = 0;

            if (source_weight != repaired_weight)
            {
                foreach (string barrierName in assignments.Keys)
                {
                    Barrier barrier = ProgramMetadata.Barriers[barrierName];
                    List<Location> locations = ProgramMetadata.Locations[barrier.SourceLocation];

                    string location = ToString(locations);
                    if (!barrier.Generated && !assignments[barrierName])
                    {
                        string message = string.Format(
                            "Remove the {0}barrier at location {1}.",
                            barrier.GridLevel ? "grid-level " : string.Empty,
                            location);

                        lines.Add(message);
                        changes.Add(location);

                        if (locations.Count > 1)
                            solutionCalls++;
                    }
                    else if (barrier.Generated && assignments[barrierName])
                    {
                        string loopMessage =
                            ProgramMetadata.LoopBarriers.SelectMany(x => x.Value).Any(x => x.Name == barrierName) ?
                            barrier.LoopDepth == 0 ? " outside the loop" : " inside the loop" : string.Empty;

                        string message = string.Format(
                            "Add a {0}barrier at location {1}{2}.",
                            barrier.GridLevel ? "grid-level " : string.Empty,
                            location, loopMessage);

                        lines.Add(message);
                        changes.Add(location);

                        solutionCalls += locations.Count > 1 ? 1 : 0;
                        solutionGrid += barrier.GridLevel ? 1 : 0;
                        solutionLoop += barrier.LoopDepth > 0 ? 1 : 0;
                    }
                }
            }

            Logger.Log($"SolutionCalls;{solutionCalls}");
            Logger.Log($"SolutionGrid;{solutionGrid}");
            Logger.Log($"SolutionLoop;{solutionLoop}");

            if (lines.Any())
                File.AppendAllLines(filename, lines.Distinct());

            return changes.Distinct();
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
