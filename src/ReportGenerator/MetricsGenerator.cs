namespace GPURepair.ReportGenerator
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using GPURepair.ReportGenerator.Records;

    public class MetricsGenerator
    {
        private string directory;

        private string timeFile;

        private string summaryFile;

        public MetricsGenerator(string directory, string timeFile, string summaryFile)
        {
            this.directory = directory;
            this.timeFile = timeFile;
            this.summaryFile = summaryFile;
        }

        public async Task Generate()
        {
            List<GPURepairTimeRecord> timings = CsvWrapper.Read<GPURepairTimeRecord>(timeFile);
            List<GPURepairRecord> records = new List<GPURepairRecord>();

            foreach (GPURepairTimeRecord timing in timings)
            {
                timing.Kernel = Common.StandardizeKernelName(timing.Kernel);
                records.Add(new GPURepairRecord
                {
                    Kernel = timing.Kernel,
                    Result = timing.Result,
                    Clang = timing.Clang,
                    Opt = timing.Opt,
                    Bugle = timing.Bugle,
                    Instrumentation = timing.Instrumentation,
                    VCGen = timing.VCGen,
                    Cruncher = timing.Cruncher,
                    Repair = timing.Repair,
                    Total = timing.Total,
                });
            }

            IEnumerable<string> directories = Directory.EnumerateDirectories(directory, "*", SearchOption.AllDirectories);
            foreach (string directory in directories)
            {
                GPURepairRecord record = records.FirstOrDefault(x => x.Kernel == Common.StandardizeKernelName(directory));
                if (record == null)
                    continue;

                foreach (string file in Directory.GetFiles(directory))
                {
                    if (file.Contains("metrics.log"))
                    {
                        string[] lines = File.ReadAllLines(file);
                        foreach (string line in lines)
                        {
                            string[] array = line.Split(new char[] { ';' });

                            if (array[0] == "Blocks")
                                record.Blocks = int.Parse(array[1]);
                            else if (array[0] == "Commands")
                                record.Commands = int.Parse(array[1]);
                            else if (array[0] == "CallCommands")
                                record.CallCommands = int.Parse(array[1]);
                            else if (array[0] == "Barriers")
                                record.Barriers = int.Parse(array[1]);
                            else if (array[0] == "GridLevelBarriers")
                                record.GridLevelBarriers = int.Parse(array[1]);
                            else if (array[0] == "LoopBarriers")
                                record.LoopBarriers = int.Parse(array[1]);
                            else if (array[0] == "Changes")
                                record.Changes = int.Parse(array[1]);
                            else if (array[0] == "SolutionCalls")
                                record.SolutionCalls = int.Parse(array[1]);
                            else if (array[0] == "SolutionGrid")
                                record.SolutionGrid = int.Parse(array[1]);
                            else if (array[0] == "SolutionLoop")
                                record.SolutionLoop = int.Parse(array[1]);
                            else if (array[0] == "RunsAfterOpt")
                                record.RunsAfterOpt = record.RunsAfterOpt + 1;
                            else if (array[0] == "FailsAfterOpt")
                                record.FailsAfterOpt = record.FailsAfterOpt + 1;
                            else if (array[0] == "ExceptionMessage")
                                record.ExceptionMessage = array[1];
                            else if (array[0] == "SourceWeight")
                                record.SourceWeight = int.Parse(array[1]);
                            else if (array[0] == "RepairedWeight")
                                record.RepairedWeight = int.Parse(array[1]);
                            else if (array[0] == "Watch")
                            {
                                if (array[1] == "MaxSAT")
                                {
                                    record.MaxSATCount = record.MaxSATCount + 1;
                                    record.MaxSATTime = record.MaxSATTime + int.Parse(array[2]);
                                }
                                else if (array[1] == "mhs")
                                {
                                    record.mhsCount = record.mhsCount + 1;
                                    record.mhsTime = record.mhsTime + int.Parse(array[2]);
                                }
                                else if (array[1] == "Verification")
                                {
                                    record.VerCount = record.VerCount + 1;
                                    record.VerTime = record.VerTime + int.Parse(array[2]);
                                }
                                else if (array[1] == "Optimization")
                                {
                                    record.OptCount = record.OptCount + 1;
                                    record.OptTime = record.OptTime + int.Parse(array[2]);
                                }
                                else if (array[1] == "SAT")
                                {
                                    record.SATCount = record.SATCount + 1;
                                    record.SATTime = record.SATTime + int.Parse(array[2]);
                                }
                            }
                        }
                    }
                    else if (file.Contains(".cu") || file.Contains(".cl") || file.Contains(".h"))
                    {
                        record.Lines += await GetLines(file);
                    }
                }
            }

            await CsvWrapper.Write(records.OrderBy(x => x.Kernel), summaryFile).ConfigureAwait(false);
        }

        private async Task<int> GetLines(string file)
        {
            int count = 0;

            string line;
            using (StreamReader reader = File.OpenText(file))
            {
                do
                {
                    line = await reader.ReadLineAsync();
                    if (!string.IsNullOrWhiteSpace(line))
                        count++;
                }
                while (line != null);
            }

            return count;
        }
    }
}
