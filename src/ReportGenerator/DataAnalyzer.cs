namespace GPURepair.ReportGenerator
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using GPURepair.ReportGenerator.Records;
    using GPURepair.ReportGenerator.Reports;

    public static class DataAnalyzer
    {
        public static IEnumerable<ToolComparisonRecord> ToolComparisonRecords { get; set; }

        public static IEnumerable<SolverComparisonRecord> SolverComparisonRecords { get; set; }

        public static IEnumerable<ConfigurationComparisonRecord> ConfigurationComparisonRecords { get; set; }

        public static void AnalyzeData(string directory)
        {
            PopulateRecords();

            StoreDate(directory, "tool_comparison.csv", ToolComparisonRecords);
            StoreDate(directory, "solver_comparison.csv", SolverComparisonRecords);
            StoreDate(directory, "configuration_comparison.csv", ConfigurationComparisonRecords);
        }

        private static IEnumerable<T> StoreDate<T>(string directory,
            string filename, IEnumerable<T> records)
        {
            string generated = directory + Path.DirectorySeparatorChar + "analysis" + Path.DirectorySeparatorChar + filename;
            CsvWrapper.Write(records, generated);

            return records;
        }

        private static void PopulateRecords()
        {
            IEnumerable<BaseReportRecord> records = GetBaseRecords();

            List<ToolComparisonRecord> toolComparisonRecords = new List<ToolComparisonRecord>();
            foreach (BaseReportRecord record in records)
                toolComparisonRecords.Add(new ToolComparisonRecord(record));
            ToolComparisonRecords = toolComparisonRecords;

            List<SolverComparisonRecord> solverComparisonRecords = new List<SolverComparisonRecord>();
            foreach (BaseReportRecord record in records)
                solverComparisonRecords.Add(new SolverComparisonRecord(record));
            SolverComparisonRecords = solverComparisonRecords;

            List<ConfigurationComparisonRecord> configurationComparisonRecords = new List<ConfigurationComparisonRecord>();
            foreach (BaseReportRecord record in records)
                configurationComparisonRecords.Add(new ConfigurationComparisonRecord(record));
            ConfigurationComparisonRecords = configurationComparisonRecords;
        }

        private static IEnumerable<BaseReportRecord> GetBaseRecords()
        {
            List<BaseReportRecord> records = new List<BaseReportRecord>();
            foreach (GPUVerifyRecord _gpuverify in FileParser.GPUVerify)
            {
                BaseReportRecord record = new BaseReportRecord();
                record.GPUVerify = _gpuverify;
                record.AutoSync = FileParser.Autosync.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);
                record.GPURepair = FileParser.GPURepair.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);
                record.GPURepair_MaxSAT = FileParser.GPURepair_MaxSAT.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);
                record.GPURepair_Grid = FileParser.GPURepair_Grid.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);
                record.GPURepair_Inspection = FileParser.GPURepair_Inspection.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);
                record.GPURepair_Grid_Inspection = FileParser.GPURepair_Grid_Inspection.FirstOrDefault(x => x.Kernel == _gpuverify.Kernel);

                records.Add(record);
            }

            return records;
        }
    }
}
