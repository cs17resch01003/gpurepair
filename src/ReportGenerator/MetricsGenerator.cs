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
                    else if (array[0] == "BarriersInsideLoop")
                        metrics.BarriersInsideLoop = int.Parse(array[1]);
                    else if (array[0] == "Changes")
                        metrics.Changes = int.Parse(array[1]);
                    else if (array[0] == "Solution_BarriersBetweenCalls")
                        metrics.Solution_BarriersBetweenCalls = int.Parse(array[1]);
                    else if (array[0] == "Solution_GridLevelBarriers")
                        metrics.Solution_GridLevelBarriers = int.Parse(array[1]);
                    else if (array[0] == "Solution_BarriersInsideLoop")
                        metrics.Solution_BarriersInsideLoop = int.Parse(array[1]);
                    else if (array[0] == "VerifierRunsAfterOptimization")
                        metrics.VerifierRunsAfterOptimization = metrics.VerifierRunsAfterOptimization + 1;
                    else if (array[0] == "VerifierFailuresAfterOptimization")
                        metrics.VerifierFailuresAfterOptimization = metrics.VerifierFailuresAfterOptimization + 1;
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
                            metrics.MaxSAT_Count = metrics.MaxSAT_Count + 1;
                            metrics.MaxSAT_Time = metrics.MaxSAT_Time + int.Parse(array[2]);
                        }
                        else if (array[1] == "MHS")
                        {
                            metrics.MHS_Count = metrics.MHS_Count + 1;
                            metrics.MHS_Time = metrics.MHS_Time + int.Parse(array[2]);
                        }
                        else if (array[1] == "Verification")
                        {
                            metrics.Verification_Count = metrics.Verification_Count + 1;
                            metrics.Verification_Time = metrics.Verification_Time + int.Parse(array[2]);
                        }
                        else if (array[1] == "Optimization")
                        {
                            metrics.Optimization_Count = metrics.Optimization_Count + 1;
                            metrics.Optimization_Time = metrics.Optimization_Time + int.Parse(array[2]);
                        }
                        else if (array[1] == "SAT")
                        {
                            metrics.SAT_Count = metrics.SAT_Count + 1;
                            metrics.SAT_Time = metrics.SAT_Time + int.Parse(array[2]);
                        }
                    }
                }

                metrics.Kernel = file.Replace(Path.DirectorySeparatorChar + "metrics.log", string.Empty);
                metrics.Kernel = metrics.Kernel.Replace(directory, string.Empty);
                metrics.Kernel = metrics.Kernel.Replace('\\', '/');

                summary.Add(metrics);
            }

            if (!summaryFilename.EndsWith(".metrics.csv"))
            {
                summaryFilename = summaryFilename.Replace(".csv", string.Empty);
                summaryFilename += ".metrics.csv";
            }

            using (StreamWriter writer = new StreamWriter(Path.Combine(directory, summaryFilename)))
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
