namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;
    using GPURepair.ReportGenerator;

    public class ToolComparisonRecord : BaseReportRecord
    {
        public ToolComparisonRecord(BaseReportRecord record) : base(record)
        {
        }

        [Name("kernel")]
        [Index(0)]
        public string Kernel => GPUVerify.Kernel;

        [Name("gv-status")]
        [Index(1)]
        public string GPUVerify_Status => GPUVerify.Result;

        [Name("as-status")]
        [Index(2)]
        public string AutoSync_Status => AutoSync.Status;

        [Name("as-time")]
        [Index(3)]
        public double AutoSync_Time => AutoSync.Time;

        [Name("as-verifier")]
        [Index(4)]
        public double AutoSync_VerCount => AutoSync.VerCount;

        [Name("gr-status")]
        [Index(5)]
        public string GPURepair_Status => GPURepair.Result(this);

        [Name("gr-time")]
        [Index(6)]
        public double GPURepair_Time => GPURepair.Total;

        [Name("gr-verifier")]
        [Index(7)]
        public double GPURepair_VerCount => GPURepair.VerCount;
    }
}
