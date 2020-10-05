namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;
    using GPURepair.ReportGenerator.Records;

    public class BaseReportRecord
    {
        public BaseReportRecord()
        {
        }

        public BaseReportRecord(BaseReportRecord record)
        {
            AutoSync = record.AutoSync ?? new AutoSyncOutRecord();
            GPUVerify = record.GPUVerify ?? new GPUVerifyRecord();
            GPURepair = record.GPURepair ?? new GPURepairRecord();
            GPURepair_MaxSAT = record.GPURepair_MaxSAT ?? new GPURepairRecord();
            GPURepair_SAT = record.GPURepair_SAT ?? new GPURepairRecord();
            GPURepair_Grid = record.GPURepair_Grid ?? new GPURepairRecord();
            GPURepair_Inspection = record.GPURepair_Inspection ?? new GPURepairRecord();
            GPURepair_Grid_Inspection = record.GPURepair_Grid_Inspection ?? new GPURepairRecord();
        }

        [Ignore]
        public AutoSyncOutRecord AutoSync { get; set; }

        [Ignore]
        public GPUVerifyRecord GPUVerify { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair_MaxSAT { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair_SAT { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair_Grid { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair_Inspection { get; set; }

        [Ignore]
        public GPURepairRecord GPURepair_Grid_Inspection { get; set; }
    }
}
