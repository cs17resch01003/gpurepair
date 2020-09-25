namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using GPURepair.ReportGenerator.Records;
    using GPURepair.ReportGenerator.Reports;

    public class SummaryGenerator
    {
        private string directory;

        private IEnumerable<AutoSyncOutRecord> autosync;

        private IEnumerable<GPUVerifyRecord> gpuverify;

        private IEnumerable<GPURepairRecord> gpurepair;

        private IEnumerable<GPURepairRecord> gpurepair_maxsat;

        private IEnumerable<GPURepairRecord> gpurepair_sat;

        private IEnumerable<GPURepairRecord> gpurepair_grid;

        private IEnumerable<GPURepairRecord> gpurepair_inspection;

        public SummaryGenerator(string directory)
        {
            this.directory = directory;
        }

        public async Task Generate()
        {
            await GenerateFiles();
            await AnalyzeData();
        }

        private async Task GenerateFiles()
        {
            autosync = await ParseSource("autosync", "autosync.csv", GetAutoSyncRecords);
            gpuverify = await ParseSource("gpuverify", "gpuverify.csv", GetGPUVerifyRecords);
            gpurepair = await ParseSource("gpurepair", "gpurepair.csv", GetGPURepairRecords);
            gpurepair_maxsat = await ParseSource("gpurepair_maxsat", "gpurepair_maxsat.csv", GetGPURepairRecords);
            gpurepair_sat = await ParseSource("gpurepair_sat", "gpurepair_sat.csv", GetGPURepairRecords);
            gpurepair_grid = await ParseSource("gpurepair_grid", "gpurepair_grid.csv", GetGPURepairRecords);
            gpurepair_inspection = await ParseSource("gpurepair_inspection", "gpurepair_inspection.csv", GetGPURepairRecords);
        }

        private async Task AnalyzeData()
        {
            await AnalyzeData("autosync-gpurepair.csv", GetToolComparisonRecords);
            await AnalyzeData("autosync-gpurepair-same.csv", GetToolComparisonSameRecords);
            await AnalyzeData("cooperative-groups.csv", GetGridComparisonRecords);
            await AnalyzeData("cooperative-groups-cuda.csv", GetGridComparisonSameRecords);
            await AnalyzeData("solver-comparison.csv", GetSolverComparisonRecords);
            await AnalyzeData("inspection-comparison.csv", GetInspectionComparisonRecords);
        }

        private async Task<IEnumerable<T>> ParseSource<T>(string folder, string filename, Func<string, IEnumerable<T>> method)
        {
            string source = directory + Path.DirectorySeparatorChar + folder;
            string generated = directory + Path.DirectorySeparatorChar + "generated" + Path.DirectorySeparatorChar + filename;

            IEnumerable<T> records = method(source);
            await CsvWrapper.Write(records, generated);

            return records;
        }

        private async Task AnalyzeData<T>(string filename, Func<IEnumerable<T>> method)
        {
            string generated = directory + Path.DirectorySeparatorChar + "analysis" + Path.DirectorySeparatorChar + filename;

            IEnumerable<T> records = method();
            await CsvWrapper.Write(records, generated);
        }

        private IEnumerable<AutoSyncOutRecord> GetAutoSyncRecords(string directory)
        {
            IEnumerable<string> files = Directory.EnumerateFiles(directory, "*.csv", SearchOption.AllDirectories);
            List<AutoSyncRecord> records = new List<AutoSyncRecord>();

            // consolidate records from all the files in a single list
            foreach (string file in files)
            {
                IEnumerable<AutoSyncRecord> list = CsvWrapper.Read<AutoSyncRecord>(file, false);
                records.AddRange(list);
            }

            // standardize the kernel name
            records.ForEach(x => x.Kernel = Common.StandardizeKernelName(x.Kernel));

            List<AutoSyncOutRecord> summary = new List<AutoSyncOutRecord>();
            foreach (AutoSyncRecord record in records)
            {
                // already visited this record
                if (summary.Any(x => x.Kernel == record.Kernel))
                    continue;

                IEnumerable<AutoSyncRecord> matches = records.Where(x => x.Kernel == record.Kernel);
                summary.Add(GetAutoSyncRecord(matches));
            }

            return summary.OrderBy(x => x.Kernel);
        }

        private AutoSyncOutRecord GetAutoSyncRecord(IEnumerable<AutoSyncRecord> records)
        {
            List<AutoSyncRecord.Status> errors = new List<AutoSyncRecord.Status>
            {
                AutoSyncRecord.Status.Error,
                AutoSyncRecord.Status.FalsePositive,
                AutoSyncRecord.Status.Timeout
            };

            AutoSyncOutRecord record = new AutoSyncOutRecord();
            record.Kernel = records.First().Kernel;

            // if the runs have multiple statuses, choose the one based on priority
            AutoSyncRecord.Status status;
            if (records.Select(x => x.Result).Distinct().Count() == 1)
                status = records.First().Result;
            else
                status = records.Select(x => x.Result).OrderByDescending(x => x.Priority).First();
            records = records.Where(x => x.Result == status);

            record.Status = status.ToString();
            if (errors.Contains(status))
            {
                record.Time = 300;
                record.VerCount = 0;
                record.Changes = 0;
            }
            else
            {
                record.Time = records.Select(x => x.Time).Average();
                record.VerCount = records.Select(x => x.VerCount).Average();

                if (status == AutoSyncRecord.Status.RepairError)
                    record.Changes = 0;
                else
                    record.Changes = records.Select(x => x.Changes).Average();
            }

            return record;
        }

        private IEnumerable<GPUVerifyRecord> GetGPUVerifyRecords(string directory)
        {
            IEnumerable<string> files = Directory.EnumerateFiles(directory, "*.csv", SearchOption.AllDirectories);
            List<GPUVerifyRecord> records = new List<GPUVerifyRecord>();

            // consolidate records from all the files in a single list
            foreach (string file in files)
            {
                IEnumerable<GPUVerifyRecord> list = CsvWrapper.Read<GPUVerifyRecord>(file);
                records.AddRange(list);
            }

            // standardize the kernel name
            records.ForEach(x => x.Kernel = Common.StandardizeKernelName(x.Kernel));

            List<GPUVerifyRecord> summary = new List<GPUVerifyRecord>();
            foreach (GPUVerifyRecord record in records)
            {
                // already visited this record
                if (summary.Any(x => x.Kernel == record.Kernel))
                    continue;

                IEnumerable<GPUVerifyRecord> matches = records.Where(x => x.Kernel == record.Kernel);
                summary.Add(GetGPUVerifyRecord(matches));
            }

            // adding the entries which fail with a command-line error
            summary.Add(new GPUVerifyRecord() { Kernel = "OpenCL/fail_equality_and_adversarial", Result = "FAIL(1)" });
            summary.Add(new GPUVerifyRecord() { Kernel = "OpenCL/global_size/local_size_fail_divide_global_size", Result = "FAIL(1)" });
            summary.Add(new GPUVerifyRecord() { Kernel = "OpenCL/global_size/mismatch_dims", Result = "FAIL(1)" });
            summary.Add(new GPUVerifyRecord() { Kernel = "OpenCL/global_size/num_groups_and_global_size", Result = "FAIL(1)" });

            return summary.OrderBy(x => x.Kernel);
        }

        private GPUVerifyRecord GetGPUVerifyRecord(IEnumerable<GPUVerifyRecord> records)
        {
            GPUVerifyRecord record = new GPUVerifyRecord();
            record.Kernel = records.First().Kernel;

            // if the runs have multiple statuses, choose the one based on priority
            GPUVerifyRecord.Status status;
            if (records.Select(x => x.ResultEnum).Distinct().Count() == 1)
                status = records.First().ResultEnum;
            else
                status = records.Select(x => x.ResultEnum).OrderByDescending(x => x.Priority).First();

            records = records.Where(x => x.ResultEnum == status);

            record.Result = status.ToString();
            record.Clang = records.Select(x => x.Clang).Average();
            record.Opt = records.Select(x => x.Opt).Average();
            record.Bugle = records.Select(x => x.Bugle).Average();
            record.VCGen = records.Select(x => x.VCGen).Average();
            record.Cruncher = records.Select(x => x.Cruncher).Average();
            record.BoogieDriver = records.Select(x => x.BoogieDriver).Average();
            record.Total = records.Select(x => x.Total).Average();

            return record;
        }

        private IEnumerable<GPURepairRecord> GetGPURepairRecords(string directory)
        {
            IEnumerable<string> files = Directory.EnumerateFiles(directory, "*.metrics.csv", SearchOption.AllDirectories);
            List<GPURepairRecord> records = new List<GPURepairRecord>();

            // consolidate records from all the files in a single list

            foreach (string file in files)
            {
                IEnumerable<GPURepairRecord> list = CsvWrapper.Read<GPURepairRecord>(file);
                records.AddRange(list);
            }

            // standardize the kernel name
            records.ForEach(x => x.Kernel = Common.StandardizeKernelName(x.Kernel));

            List<GPURepairRecord> summary = new List<GPURepairRecord>();
            foreach (GPURepairRecord record in records)
            {
                // already visited this record
                if (summary.Any(x => x.Kernel == record.Kernel))
                    continue;

                IEnumerable<GPURepairRecord> matches = records.Where(x => x.Kernel == record.Kernel);
                summary.Add(GetGPURepairRecord(matches));
            }

            // adding the entries which fail with a command-line error
            summary.Add(new GPURepairRecord() { Kernel = "OpenCL/fail_equality_and_adversarial", Result = "FAIL(1)" });
            summary.Add(new GPURepairRecord() { Kernel = "OpenCL/global_size/local_size_fail_divide_global_size", Result = "FAIL(1)" });
            summary.Add(new GPURepairRecord() { Kernel = "OpenCL/global_size/mismatch_dims", Result = "FAIL(1)" });
            summary.Add(new GPURepairRecord() { Kernel = "OpenCL/global_size/num_groups_and_global_size", Result = "FAIL(1)" });

            return summary.OrderBy(x => x.Kernel);
        }

        private GPURepairRecord GetGPURepairRecord(IEnumerable<GPURepairRecord> records)
        {
            GPURepairRecord record = new GPURepairRecord();
            record.Kernel = records.First().Kernel;

            // if the runs have multiple statuses, choose the one based on priority
            GPURepairTimeRecord.Status status;
            if (records.Select(x => x.ResultEnum).Distinct().Count() == 1)
                status = records.First().ResultEnum;
            else
                status = records.Select(x => x.ResultEnum).OrderByDescending(x => x.Priority).First();

            records = records.Where(x => x.ResultEnum == status);

            record.Result = status.ToString();
            record.Clang = records.Select(x => x.Clang).Average();
            record.Opt = records.Select(x => x.Opt).Average();
            record.Bugle = records.Select(x => x.Bugle).Average();
            record.Instrumentation = records.Select(x => x.Instrumentation).Average();
            record.VCGen = records.Select(x => x.VCGen).Average();
            record.Cruncher = records.Select(x => x.Cruncher).Average();
            record.Repair = records.Select(x => x.Repair).Average();
            record.Total = records.Select(x => x.Total).Average();
            record.Lines = records.Select(x => x.Lines).Average();
            record.Blocks = records.Select(x => x.Blocks).Average();
            record.Commands = records.Select(x => x.Commands).Average();
            record.CallCommands = records.Select(x => x.CallCommands).Average();
            record.Barriers = records.Select(x => x.Barriers).Average();
            record.GridLevelBarriers = records.Select(x => x.GridLevelBarriers).Average();
            record.LoopBarriers = records.Select(x => x.LoopBarriers).Average();
            record.Changes = records.Select(x => x.Changes).Average();
            record.SolutionCalls = records.Select(x => x.SolutionCalls).Average();
            record.SolutionGrid = records.Select(x => x.SolutionGrid).Average();
            record.SolutionLoop = records.Select(x => x.SolutionLoop).Average();
            record.SourceWeight = records.Select(x => x.SourceWeight).Average();
            record.RepairedWeight = records.Select(x => x.RepairedWeight).Average();
            record.RunsAfterOpt = records.Select(x => x.RunsAfterOpt).Average();
            record.FailsAfterOpt = records.Select(x => x.FailsAfterOpt).Average();
            record.mhsCount = records.Select(x => x.mhsCount).Average();
            record.mhsTime = records.Select(x => x.mhsTime).Average();
            record.MaxSATCount = records.Select(x => x.MaxSATCount).Average();
            record.MaxSATTime = records.Select(x => x.MaxSATTime).Average();
            record.SATCount = records.Select(x => x.SATCount).Average();
            record.SATTime = records.Select(x => x.SATTime).Average();
            record.VerCount = records.Select(x => x.VerCount).Average();
            record.VerTime = records.Select(x => x.VerTime).Average();
            record.OptCount = records.Select(x => x.OptCount).Average();
            record.OptTime = records.Select(x => x.OptTime).Average();
            record.ExceptionMessage = string.Join(";", records.Select(x => x.ExceptionMessage).Distinct());

            return record;
        }

        private IEnumerable<ToolComparisonRecord> GetToolComparisonRecords()
        {
            List<ToolComparisonRecord> records = new List<ToolComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in gpuverify)
            {
                records.Add(new ToolComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (AutoSyncOutRecord _autoSync in autosync)
            {
                ToolComparisonRecord record = records.First(x => x.Kernel == _autoSync.Kernel);
                record.AS_Status = _autoSync.Status;
                record.AS_Time = _autoSync.Time;
                record.AS_VerCount = _autoSync.VerCount;
                record.AS_Changes = _autoSync.Changes;
            }

            foreach (GPURepairRecord _gpurepair in gpurepair)
            {
                ToolComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.GR_Status = _gpurepair.Result;
                record.GR_Time = _gpurepair.Total;
                record.GR_VerCount = _gpurepair.VerCount;
                record.GR_Changes = _gpurepair.Changes;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private IEnumerable<ToolComparisonRecord> GetToolComparisonSameRecords()
        {
            IEnumerable<ToolComparisonRecord> records = GetToolComparisonRecords();
            IEnumerable<ToolComparisonRecord> unchanged = records.Where(
                x => x.AS_Status == "UNCHANGED" && x.GR_Status == "PASS" && x.GR_Changes == 0);
            IEnumerable<ToolComparisonRecord> repaired = records.Where(
                x => x.AS_Status == "REPAIRED" && x.GR_Status == "PASS");

            return unchanged.Union(repaired).OrderBy(x => x.Kernel);
        }

        private IEnumerable<GridComparisonRecord> GetGridComparisonRecords()
        {
            List<GridComparisonRecord> records = new List<GridComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in gpuverify)
            {
                records.Add(new GridComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (GPURepairRecord _gpurepair in gpurepair)
            {
                GridComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.Enabled_Status = _gpurepair.Result;
                record.Enabled_Time = _gpurepair.Total;
                record.Enabled_SolverCount = _gpurepair.SolverCount;
                record.Enabled_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in gpurepair_grid)
            {
                GridComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.Disabled_Status = _gpurepair.Result;
                record.Disabled_Time = _gpurepair.Total;
                record.Disabled_SolverCount = _gpurepair.SolverCount;
                record.Disabled_VerCount = _gpurepair.VerCount;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private IEnumerable<GridComparisonRecord> GetGridComparisonSameRecords()
        {
            IEnumerable<GridComparisonRecord> records = GetGridComparisonRecords();
            return records.Where(x => x.Enabled_Status == x.Disabled_Status && !x.Kernel.Contains("OpenCL")).OrderBy(x => x.Kernel);
        }

        private IEnumerable<SolverComparisonRecord> GetSolverComparisonRecords()
        {
            List<SolverComparisonRecord> records = new List<SolverComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in gpuverify)
            {
                records.Add(new SolverComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (GPURepairRecord _gpurepair in gpurepair)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.mhs_Status = _gpurepair.Result;
                record.mhs_Time = _gpurepair.Total;
                record.mhs_SolverCount = _gpurepair.SolverCount;
                record.mhs_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in gpurepair_maxsat)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.MaxSAT_Status = _gpurepair.Result;
                record.MaxSAT_Time = _gpurepair.Total;
                record.MaxSAT_SolverCount = _gpurepair.SolverCount;
                record.MaxSAT_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in gpurepair_sat)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.SAT_Status = _gpurepair.Result;
                record.SAT_Time = _gpurepair.Total;
                record.SAT_SolverCount = _gpurepair.SolverCount;
                record.SAT_VerCount = _gpurepair.VerCount;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private IEnumerable<InspectionComparisonRecord> GetInspectionComparisonRecords()
        {
            List<InspectionComparisonRecord> records = new List<InspectionComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in gpuverify)
            {
                records.Add(new InspectionComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (GPURepairRecord _gpurepair in gpurepair)
            {
                InspectionComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.Enabled_Status = _gpurepair.Result;
                record.Enabled_Time = _gpurepair.Total;
                record.Enabled_SolverCount = _gpurepair.SolverCount;
                record.Enabled_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in gpurepair_inspection)
            {
                InspectionComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.Disabled_Status = _gpurepair.Result;
                record.Disabled_Time = _gpurepair.Total;
                record.Disabled_SolverCount = _gpurepair.SolverCount;
                record.Disabled_VerCount = _gpurepair.VerCount;
            }

            return records.OrderBy(x => x.Kernel);
        }
    }
}
