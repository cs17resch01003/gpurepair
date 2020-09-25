namespace GPURepair.ReportGenerator
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using GPURepair.ReportGenerator.Records;

    public class SummaryGenerator
    {
        private string directory;

        public SummaryGenerator(string directory)
        {
            this.directory = directory;
        }

        public async Task Generate()
        {
            IEnumerable<AutoSyncOutRecord> autosync = GetAutoSyncRecords(
                directory + Path.DirectorySeparatorChar + "autosync");
            await CsvWrapper.Write(autosync.OrderBy(x => x.Kernel),
                directory + Path.DirectorySeparatorChar + "generated" + Path.DirectorySeparatorChar + "autosync.csv").ConfigureAwait(false);

            IEnumerable<GPUVerifyRecord> gpuverify = GetGPUVerifyRecords(
                directory + Path.DirectorySeparatorChar + "gpuverify");
            await CsvWrapper.Write(gpuverify.OrderBy(x => x.Kernel),
                directory + Path.DirectorySeparatorChar + "generated" + Path.DirectorySeparatorChar + "gpuverify.csv").ConfigureAwait(false);
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

            return summary;
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
                record.VerCount = null;
                record.Changes = null;
            }
            else
            {
                record.Time = records.Select(x => x.Time).Average();
                record.VerCount = records.Select(x => x.VerCount).Average();

                if (status == AutoSyncRecord.Status.RepairError)
                    record.Changes = null;
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

            return summary;
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
    }
}
