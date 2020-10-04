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

        [Name("as-status")]
        [Index(2)]
        public string AutoSync_Status { get; set; }

        [Name("as-time")]
        [Index(3)]
        public double AutoSync_Time { get; set; }

        [Name("as-ver-count")]
        [Index(4)]
        public double AutoSync_VerCount { get; set; }

        [Name("def-status")]
        [Index(5)]
        public string Default_Status { get; set; }

        [Name("def-time")]
        [Index(6)]
        public double Default_Time { get; set; }

        [Name("def-solver-count")]
        [Index(7)]
        public double Default_SolverCount { get; set; }

        [Name("def-ver-count")]
        [Index(8)]
        public double Default_VerCount { get; set; }

        [Name("def-changes")]
        [Index(9)]
        public double Default_Changes { get; set; }

        [Name("disgrid-status")]
        [Index(10)]
        public string DG_Status { get; set; }

        [Name("disgrid-time")]
        [Index(11)]
        public double DG_Time { get; set; }

        [Name("disgrid-solver-count")]
        [Index(12)]
        public double DG_SolverCount { get; set; }

        [Name("disgrid-ver-count")]
        [Index(13)]
        public double DG_VerCount { get; set; }

        [Name("disgrid-changes")]
        [Index(14)]
        public double DG_Changes { get; set; }

        [Name("disinspect-status")]
        [Index(15)]
        public string DI_Status { get; set; }

        [Name("disinspect-time")]
        [Index(16)]
        public double DI_Time { get; set; }

        [Name("disinspect-solver-count")]
        [Index(17)]
        public double DI_SolverCount { get; set; }

        [Name("disinspect-ver-count")]
        [Index(18)]
        public double DI_VerCount { get; set; }

        [Name("disinspect-changes")]
        [Index(19)]
        public double DI_Changes { get; set; }

        [Name("disgrid-disinspect-status")]
        [Index(20)]
        public string DG_DI_Status { get; set; }

        [Name("disgrid-disinspect-time")]
        [Index(21)]
        public double DG_DI_Time { get; set; }

        [Name("disgrid-disinspect-solver-count")]
        [Index(22)]
        public double DG_DI_SolverCount { get; set; }

        [Name("disgrid-disinspect-ver-count")]
        [Index(23)]
        public double DG_DI_VerCount { get; set; }

        [Name("disgrid-disinspect-changes")]
        [Index(24)]
        public double DG_DI_Changes { get; set; }
    }
}
