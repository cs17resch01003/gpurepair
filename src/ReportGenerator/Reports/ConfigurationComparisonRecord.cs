namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;

    public class ConfigurationComparisonRecord
    {
        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("gv-status")]
        [Index(1)]
        public string GV_Status { get; set; }

        [Name("def-status")]
        [Index(2)]
        public string Default_Status { get; set; }

        [Name("def-time")]
        [Index(3)]
        public double Default_Time { get; set; }

        [Name("def-solver-count")]
        [Index(4)]
        public double Default_SolverCount { get; set; }

        [Name("def-ver-count")]
        [Index(5)]
        public double Default_VerCount { get; set; }

        [Name("disgrid-status")]
        [Index(6)]
        public string DG_Status { get; set; }

        [Name("disgrid-time")]
        [Index(7)]
        public double DG_Time { get; set; }

        [Name("disgrid-solver-count")]
        [Index(8)]
        public double DG_SolverCount { get; set; }

        [Name("disgrid-ver-count")]
        [Index(9)]
        public double DG_VerCount { get; set; }

        [Name("disinspect-status")]
        [Index(10)]
        public string DI_Status { get; set; }

        [Name("disinspect-time")]
        [Index(11)]
        public double DI_Time { get; set; }

        [Name("disinspect-solver-count")]
        [Index(12)]
        public double DI_SolverCount { get; set; }

        [Name("disinspect-ver-count")]
        [Index(13)]
        public double DI_VerCount { get; set; }

        [Name("disgrid-disinspect-status")]
        [Index(14)]
        public string DG_DI_Status { get; set; }

        [Name("disgrid-disinspect-time")]
        [Index(15)]
        public double DG_DI_Time { get; set; }

        [Name("disgrid-disinspect-solver-count")]
        [Index(16)]
        public double DG_DI_SolverCount { get; set; }

        [Name("disgrid-disinspect-ver-count")]
        [Index(17)]
        public double DG_DI_VerCount { get; set; }
    }
}
