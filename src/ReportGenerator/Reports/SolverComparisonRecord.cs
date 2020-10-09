namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;
    using GPURepair.ReportGenerator;

    public class SolverComparisonRecord : BaseReportRecord
    {
        public SolverComparisonRecord(BaseReportRecord record) : base(record)
        {
        }

        [Name("kernel")]
        [Index(0)]
        public string Kernel => GPUVerify.Kernel;

        [Name("gv-status")]
        [Index(1)]
        public string GPUVerify_Status => GPUVerify.Result;

        [Name("mhs-status")]
        [Index(2)]
        public string mhs_Status => GPURepair.Result(this);

        [Name("mhs-time")]
        [Index(3)]
        public double mhs_Time => GPURepair.Total;

        [Name("mhs-solver")]
        [Index(4)]
        public double mhs_SolverCount => GPURepair.SolverCount;

        [Name("mhs-verifier")]
        [Index(5)]
        public double mhs_VerCount => GPURepair.VerCount;

        [Name("maxsat-status")]
        [Index(6)]
        public string MaxSAT_Status => GPURepair_MaxSAT.Result(this);

        [Name("maxsat-time")]
        [Index(7)]
        public double MaxSAT_Time => GPURepair_MaxSAT.Total;

        [Name("maxsat-solver")]
        [Index(8)]
        public double MaxSAT_SolverCount => GPURepair_MaxSAT.SolverCount;

        [Name("maxsat-verifier")]
        [Index(9)]
        public double MaxSAT_VerCount => GPURepair_MaxSAT.VerCount;
    }
}
