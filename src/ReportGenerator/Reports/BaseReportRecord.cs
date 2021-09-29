namespace GPURepair.ReportGenerator.Reports
{
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
            GPURepair_Grid = record.GPURepair_Grid ?? new GPURepairRecord();
            GPURepair_Inspection = record.GPURepair_Inspection ?? new GPURepairRecord();
            GPURepair_Grid_Inspection = record.GPURepair_Grid_Inspection ?? new GPURepairRecord();
            GPURepair_Axioms = record.GPURepair_Axioms ?? new GPURepairRecord();
            GPURepair_Classic = record.GPURepair_Classic ?? new GPURepairRecord();
        }

        public AutoSyncOutRecord AutoSync;

        public GPUVerifyRecord GPUVerify;

        public GPURepairRecord GPURepair;

        public GPURepairRecord GPURepair_MaxSAT;

        public GPURepairRecord GPURepair_Grid;

        public GPURepairRecord GPURepair_Inspection;

        public GPURepairRecord GPURepair_Grid_Inspection;

        public GPURepairRecord GPURepair_Axioms;

        public GPURepairRecord GPURepair_Classic;
    }
}
