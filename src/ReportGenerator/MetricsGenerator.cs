namespace ReportGenerator
{
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Threading.Tasks;
    using CsvHelper;

    public class MetricsGenerator
    {
        private string directory;

        private string summaryFilename;

        public MetricsGenerator(string directory, string summaryFilename)
        {
            this.directory = directory;
            this.summaryFilename = summaryFilename;
        }

        public async Task Generate()
        {
            List<Metrics> summary = new List<Metrics>();

            IEnumerable<string> files = Directory.EnumerateFiles(directory, "metrics.log", SearchOption.AllDirectories);
            foreach (string file in files)
            {
                Metrics metrics = new Metrics();

                string[] lines = File.ReadAllLines(file);
                foreach (string line in lines)
                {
                    string[] array = line.Split(new char[] { ';' });

                    if (array[0] == "Blocks")
                        metrics.Blocks = int.Parse(array[1]);
                    else if (array[0] == "Commands")
                        metrics.Commands = int.Parse(array[1]);
                    else if (array[0] == "CallCommands")
                        metrics.CallCommands = int.Parse(array[1]);
                    else if (array[0] == "Barriers")
                        metrics.Barriers = int.Parse(array[1]);
                    else if (array[0] == "GridLevelBarriers")
                        metrics.GridLevelBarriers = int.Parse(array[1]);
                    else if (array[0] == "LoopBarriers")
                        metrics.LoopBarriers = int.Parse(array[1]);
                    else if (array[0] == "Changes")
                        metrics.Changes = int.Parse(array[1]);
                    else if (array[0] == "SolutionCalls")
                        metrics.SolutionCalls = int.Parse(array[1]);
                    else if (array[0] == "SolutionGrid")
                        metrics.SolutionGrid = int.Parse(array[1]);
                    else if (array[0] == "SolutionLoop")
                        metrics.SolutionLoop = int.Parse(array[1]);
                    else if (array[0] == "RunsAfterOpt")
                        metrics.RunsAfterOpt = metrics.RunsAfterOpt + 1;
                    else if (array[0] == "FailsAfterOpt")
                        metrics.FailsAfterOpt = metrics.FailsAfterOpt + 1;
                    else if (array[0] == "ExceptionMessage")
                        metrics.ExceptionMessage = array[1];
                    else if (array[0] == "SourceWeight")
                        metrics.SourceWeight = int.Parse(array[1]);
                    else if (array[0] == "RepairedWeight")
                        metrics.RepairedWeight = int.Parse(array[1]);
                    else if (array[0] == "Watch")
                    {
                        if (array[1] == "MaxSAT")
                        {
                            metrics.MaxSATCount = metrics.MaxSATCount + 1;
                            metrics.MaxSATTime = metrics.MaxSATTime + int.Parse(array[2]);
                        }
                        else if (array[1] == "mhs")
                        {
                            metrics.mhsCount = metrics.mhsCount + 1;
                            metrics.mhsTime = metrics.mhsTime + int.Parse(array[2]);
                        }
                        else if (array[1] == "Verification")
                        {
                            metrics.VerCount = metrics.VerCount + 1;
                            metrics.VerTime = metrics.VerTime + int.Parse(array[2]);
                        }
                        else if (array[1] == "Optimization")
                        {
                            metrics.OptCount = metrics.OptCount + 1;
                            metrics.OptTime = metrics.OptTime + int.Parse(array[2]);
                        }
                        else if (array[1] == "SAT")
                        {
                            metrics.SATCount = metrics.SATCount + 1;
                            metrics.SATTime = metrics.SATTime + int.Parse(array[2]);
                        }
                    }
                }

                metrics.Kernel = file.Replace(Path.DirectorySeparatorChar + "metrics.log", string.Empty);
                metrics.Kernel = metrics.Kernel.Replace(directory, string.Empty);
                metrics.Kernel = metrics.Kernel.Replace('\\', '/');

                summary.Add(metrics);
            }

            using (StreamWriter writer = new StreamWriter(summaryFilename))
            using (CsvWriter csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.Configuration.ShouldQuote = (field, context) =>
                {
                    return true;
                };

                await csv.WriteRecordsAsync(summary).ConfigureAwait(false);
            }
        }
    }
}
