namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using GPURepair.ReportGenerator.Records;

    public static class FileParser
    {
        public static IEnumerable<AutoSyncOutRecord> Autosync { get; set; }

        public static IEnumerable<GPUVerifyRecord> GPUVerify { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair_MaxSAT { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair_SAT { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair_Grid { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair_Inspection { get; set; }

        public static IEnumerable<GPURepairRecord> GPURepair_Grid_Inspection { get; set; }

        public static async Task ParseFiles(string directory)
        {
            Autosync = await ParseSource(directory, "autosync", "autosync.csv", GetAutoSyncRecords);
            GPUVerify = await ParseSource(directory, "gpuverify", "gpuverify.csv", GetGPUVerifyRecords);
            GPURepair = await ParseSource(directory, "gpurepair", "gpurepair.csv", GetGPURepairRecords);
            GPURepair_MaxSAT = await ParseSource(directory, "gpurepair_maxsat", "gpurepair_maxsat.csv", GetGPURepairRecords);
            GPURepair_SAT = await ParseSource(directory, "gpurepair_sat", "gpurepair_sat.csv", GetGPURepairRecords);
            GPURepair_Grid = await ParseSource(directory, "gpurepair_grid", "gpurepair_grid.csv", GetGPURepairRecords);
            GPURepair_Inspection = await ParseSource(directory, "gpurepair_inspection", "gpurepair_inspection.csv", GetGPURepairRecords);
            GPURepair_Grid_Inspection = await ParseSource(directory, "gpurepair_grid_inspection", "gpurepair_grid_inspection.csv", GetGPURepairRecords);
        }

        private static async Task<IEnumerable<T>> ParseSource<T>(
            string directory, string folder, string filename, Func<string, IEnumerable<T>> method)
        {
            string source = directory + Path.DirectorySeparatorChar + folder;
            string generated = directory + Path.DirectorySeparatorChar + "generated" + Path.DirectorySeparatorChar + filename;

            IEnumerable<T> records = method(source);
            await CsvWrapper.Write(records, generated);

            return records;
        }

        private static IEnumerable<AutoSyncOutRecord> GetAutoSyncRecords(string directory)
        {
            if (!Directory.Exists(directory))
                return new List<AutoSyncOutRecord>();

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

        private static AutoSyncOutRecord GetAutoSyncRecord(IEnumerable<AutoSyncRecord> records)
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

        private static IEnumerable<GPUVerifyRecord> GetGPUVerifyRecords(string directory)
        {
            if (!Directory.Exists(directory))
                return new List<GPUVerifyRecord>();

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

        private static GPUVerifyRecord GetGPUVerifyRecord(IEnumerable<GPUVerifyRecord> records)
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

        private static IEnumerable<GPURepairRecord> GetGPURepairRecords(string directory)
        {
            if (!Directory.Exists(directory))
                return new List<GPURepairRecord>();

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

        private static GPURepairRecord GetGPURepairRecord(IEnumerable<GPURepairRecord> records)
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
    }
}
