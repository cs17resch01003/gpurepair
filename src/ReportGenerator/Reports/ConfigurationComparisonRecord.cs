namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;
    using GPURepair.ReportGenerator;

    public class ConfigurationComparisonRecord : BaseReportRecord
    {
        public ConfigurationComparisonRecord(BaseReportRecord record) : base(record)
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

        [Name("gr-solver")]
        [Index(7)]
        public double GPURepair_SolverCount => GPURepair.SolverCount;

        [Name("gr-verifier")]
        [Index(8)]
        public double GPURepair_VerCount => GPURepair.VerCount;

        [Name("maxsat-status")]
        [Index(9)]
        public string MaxSAT_Status => GPURepair_MaxSAT.Result(this);

        [Name("maxsat-time")]
        [Index(10)]
        public double MaxSAT_Time => GPURepair_MaxSAT.Total;

        [Name("maxsat-solver")]
        [Index(11)]
        public double MaxSAT_SolverCount => GPURepair_MaxSAT.SolverCount;

        [Name("maxsat-verifier")]
        [Index(12)]
        public double MaxSAT_VerCount => GPURepair_MaxSAT.VerCount;

        [Name("grid-status")]
        [Index(13)]
        public string Grid_Status => GPURepair_Grid.Result(this);

        [Name("grid-time")]
        [Index(14)]
        public double Grid_Time => GPURepair_Grid.Total;

        [Name("grid-solver")]
        [Index(15)]
        public double Grid_SolverCount => GPURepair_Grid.SolverCount;

        [Name("grid-verifier")]
        [Index(16)]
        public double Grid_VerCount => GPURepair_Grid.VerCount;

        [Name("ins-status")]
        [Index(17)]
        public string Inspection_Status => GPURepair_Inspection.Result(this);

        [Name("ins-time")]
        [Index(18)]
        public double Inspection_Time => GPURepair_Inspection.Total;

        [Name("ins-solver")]
        [Index(19)]
        public double Inspection_SolverCount => GPURepair_Inspection.SolverCount;

        [Name("ins-verifier")]
        [Index(20)]
        public double Inspection_VerCount => GPURepair_Inspection.VerCount;

        [Name("grid-ins-status")]
        [Index(21)]
        public string Grid_Inspection_Status => GPURepair_Grid_Inspection.Result(this);

        [Name("grid-ins-time")]
        [Index(22)]
        public double Grid_Inspection_Time => GPURepair_Grid_Inspection.Total;

        [Name("grid-ins-solver")]
        [Index(23)]
        public double Grid_Inspection_SolverCount => GPURepair_Grid_Inspection.SolverCount;

        [Name("grid-ins-verifier")]
        [Index(24)]
        public double Grid_Inspection_VerCount => GPURepair_Grid_Inspection.VerCount;

        [Name("axioms-status")]
        [Index(25)]
        public string Axioms_Status => GPURepair_Axioms.Result(this);

        [Name("axioms-time")]
        [Index(26)]
        public double Axioms_Time => GPURepair_Axioms.Total;

        [Name("axioms-solver")]
        [Index(27)]
        public double Axioms_SolverCount => GPURepair_Axioms.SolverCount;

        [Name("axioms-verifier")]
        [Index(28)]
        public double Axioms_VerCount => GPURepair_Axioms.VerCount;

        [Name("classic-status")]
        [Index(29)]
        public string Classic_Status => GPURepair_Classic.Result(this);

        [Name("classic-time")]
        [Index(30)]
        public double Classic_Time => GPURepair_Classic.Total;

        [Name("classic-solver")]
        [Index(31)]
        public double Classic_SolverCount => GPURepair_Classic.SolverCount;

        [Name("classic-verifier")]
        [Index(32)]
        public double Classic_VerCount => GPURepair_Classic.VerCount;
    }
}
