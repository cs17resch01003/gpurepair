namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using GPURepair.ReportGenerator.Records;
    using GPURepair.ReportGenerator.Reports;

    public static class DataAnalyzer
    {
        public static IEnumerable<ToolComparisonRecord> ToolComparisonRecords { get; set; }

        public static IEnumerable<ToolComparisonRecord> ToolComparisonSameRecords { get; set; }

        public static IEnumerable<SolverComparisonRecord> SolverComparisonRecords { get; set; }

        public static IEnumerable<ConfigurationComparisonRecord> ConfigurationComparisonRecords { get; set; }

        public static IEnumerable<ConfigurationComparisonRecord> ConfigurationComparisonSameRecords { get; set; }

        public static async Task AnalyzeData(string directory)
        {
            ToolComparisonRecords = await AnalyzeData(directory, "tool_comparison.csv", GetToolComparisonRecords);
            ToolComparisonSameRecords = await AnalyzeData(directory, "tool_comparison_same.csv", GetToolComparisonSameRecords);
            SolverComparisonRecords = await AnalyzeData(directory, "solver_comparison.csv", GetSolverComparisonRecords);
            ConfigurationComparisonRecords = await AnalyzeData(directory, "configuration_comparison.csv", GetConfigurationComparisonRecords);
            ConfigurationComparisonSameRecords = await AnalyzeData(directory, "configuration_comparison_same.csv", GetConfigurationComparisonSameRecords);
        }

        private static async Task<IEnumerable<T>> AnalyzeData<T>(
            string directory, string filename, Func<IEnumerable<T>> method)
        {
            string generated = directory + Path.DirectorySeparatorChar + "analysis" + Path.DirectorySeparatorChar + filename;

            IEnumerable<T> records = method();
            await CsvWrapper.Write(records, generated);

            return records;
        }

        private static IEnumerable<ToolComparisonRecord> GetToolComparisonRecords()
        {
            List<ToolComparisonRecord> records = new List<ToolComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in FileParser.GPUVerify)
            {
                records.Add(new ToolComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (AutoSyncOutRecord _autoSync in FileParser.Autosync)
            {
                ToolComparisonRecord record = records.First(x => x.Kernel == _autoSync.Kernel);
                record.AS_Status = _autoSync.Status;
                record.AS_Time = _autoSync.Time;
                record.AS_VerCount = _autoSync.VerCount;
                record.AS_Changes = _autoSync.Changes;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair)
            {
                ToolComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.GR_Status = _gpurepair.Result;
                record.GR_Time = _gpurepair.Total;
                record.GR_VerCount = _gpurepair.VerCount;
                record.GR_Changes = _gpurepair.Changes;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private static IEnumerable<ToolComparisonRecord> GetToolComparisonSameRecords()
        {
            IEnumerable<ToolComparisonRecord> records = GetToolComparisonRecords();
            IEnumerable<ToolComparisonRecord> unchanged = records.Where(
                x => x.AS_Status == "UNCHANGED" && x.GR_Status == "PASS" && x.GR_Changes == 0);
            IEnumerable<ToolComparisonRecord> repaired = records.Where(
                x => x.AS_Status == "REPAIRED" && x.GR_Status == "PASS");

            return unchanged.Union(repaired).OrderBy(x => x.Kernel);
        }

        private static IEnumerable<SolverComparisonRecord> GetSolverComparisonRecords()
        {
            List<SolverComparisonRecord> records = new List<SolverComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in FileParser.GPUVerify)
            {
                records.Add(new SolverComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.mhs_Status = _gpurepair.Result;
                record.mhs_Time = _gpurepair.Total;
                record.mhs_SolverCount = _gpurepair.SolverCount;
                record.mhs_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair_MaxSAT)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.MaxSAT_Status = _gpurepair.Result;
                record.MaxSAT_Time = _gpurepair.Total;
                record.MaxSAT_SolverCount = _gpurepair.SolverCount;
                record.MaxSAT_VerCount = _gpurepair.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair_SAT)
            {
                SolverComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.SAT_Status = _gpurepair.Result;
                record.SAT_Time = _gpurepair.Total;
                record.SAT_SolverCount = _gpurepair.SolverCount;
                record.SAT_VerCount = _gpurepair.VerCount;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private static IEnumerable<ConfigurationComparisonRecord> GetConfigurationComparisonRecords()
        {
            List<ConfigurationComparisonRecord> records = new List<ConfigurationComparisonRecord>();
            foreach (GPUVerifyRecord _gpuverify in FileParser.GPUVerify)
            {
                records.Add(new ConfigurationComparisonRecord
                {
                    Kernel = _gpuverify.Kernel,
                    GV_Status = _gpuverify.Result
                });
            }

            foreach (AutoSyncOutRecord _autosync in FileParser.Autosync)
            {
                ConfigurationComparisonRecord record = records.First(x => x.Kernel == _autosync.Kernel);
                record.AutoSync_Status = _autosync.Status;
                record.AutoSync_Time = _autosync.Time;
                record.AutoSync_VerCount = _autosync.VerCount;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair)
            {
                ConfigurationComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.Default_Status = _gpurepair.Result;
                record.Default_Time = _gpurepair.Total;
                record.Default_SolverCount = _gpurepair.SolverCount;
                record.Default_VerCount = _gpurepair.VerCount;
                record.Default_Changes = _gpurepair.Changes;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair_Grid)
            {
                ConfigurationComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.DG_Status = _gpurepair.Result;
                record.DG_Time = _gpurepair.Total;
                record.DG_SolverCount = _gpurepair.SolverCount;
                record.DG_VerCount = _gpurepair.VerCount;
                record.DG_Changes = _gpurepair.Changes;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair_Inspection)
            {
                ConfigurationComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.DI_Status = _gpurepair.Result;
                record.DI_Time = _gpurepair.Total;
                record.DI_SolverCount = _gpurepair.SolverCount;
                record.DI_VerCount = _gpurepair.VerCount;
                record.DI_Changes = _gpurepair.Changes;
            }

            foreach (GPURepairRecord _gpurepair in FileParser.GPURepair_Grid_Inspection)
            {
                ConfigurationComparisonRecord record = records.First(x => x.Kernel == _gpurepair.Kernel);
                record.DG_DI_Status = _gpurepair.Result;
                record.DG_DI_Time = _gpurepair.Total;
                record.DG_DI_SolverCount = _gpurepair.SolverCount;
                record.DG_DI_VerCount = _gpurepair.VerCount;
                record.DG_DI_Changes = _gpurepair.Changes;
            }

            return records.OrderBy(x => x.Kernel);
        }

        private static IEnumerable<ConfigurationComparisonRecord> GetConfigurationComparisonSameRecords()
        {
            IEnumerable<ConfigurationComparisonRecord> records = GetConfigurationComparisonRecords();
            IEnumerable<ConfigurationComparisonRecord> same = records.Where(x =>
            {
                bool dg = x.Default_Status == x.DG_Status;
                if (dg == true && x.Default_Status == "PASS" && x.Default_Changes == 0)
                    dg = x.DG_Changes == 0;

                bool di = x.Default_Status == x.DI_Status;
                if (di == true && x.Default_Status == "PASS" && x.Default_Changes == 0)
                    di = x.DI_Changes == 0;

                bool dg_di = x.Default_Status == x.DG_DI_Status;
                if (dg_di == true && x.Default_Status == "PASS" && x.Default_Changes == 0)
                    dg_di = x.DG_DI_Changes == 0;

                return dg && di && dg_di;
            });

            return same;
        }
    }
}
