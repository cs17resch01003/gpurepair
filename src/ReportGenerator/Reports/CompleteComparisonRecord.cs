namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;

    public class CompleteComparisonRecord
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

        [Name("mhs-status")]
        [Index(3)]
        public string mhs_Status { get; set; }

        [Name("maxsat-status")]
        [Index(4)]
        public string MaxSAT_Status { get; set; }

        [Name("sat-status")]
        [Index(5)]
        public string SAT_Status { get; set; }

        [Name("disgrid-status")]
        [Index(6)]
        public string DG_Status { get; set; }

        [Name("disinspect-status")]
        [Index(7)]
        public string DI_Status { get; set; }

        [Name("disgrid-disinspect-status")]
        [Index(8)]
        public string DG_DI_Status { get; set; }

        [Name("as-time")]
        [Index(9)]
        public double AutoSync_Time { get; set; }

        [Name("mhs-time")]
        [Index(10)]
        public double mhs_Time { get; set; }

        [Name("maxsat-time")]
        [Index(11)]
        public double MaxSAT_Time { get; set; }

        [Name("sat-time")]
        [Index(12)]
        public double SAT_Time { get; set; }

        [Name("disgrid-time")]
        [Index(13)]
        public double DG_Time { get; set; }

        [Name("disinspect-time")]
        [Index(14)]
        public double DI_Time { get; set; }

        [Name("disgrid-disinspect-time")]
        [Index(15)]
        public double DG_DI_Time { get; set; }

        [Name("as-ver-count")]
        [Index(16)]
        public double AutoSync_VerCount { get; set; }

        [Name("mhs-ver-count")]
        [Index(17)]
        public double mhs_VerCount { get; set; }

        [Name("maxsat-ver-count")]
        [Index(18)]
        public double MaxSAT_VerCount { get; set; }

        [Name("sat-ver-count")]
        [Index(19)]
        public double SAT_VerCount { get; set; }

        [Name("disgrid-ver-count")]
        [Index(20)]
        public double DG_VerCount { get; set; }

        [Name("disinspect-ver-count")]
        [Index(21)]
        public double DI_VerCount { get; set; }

        [Name("disgrid-disinspect-ver-count")]
        [Index(22)]
        public double DG_DI_VerCount { get; set; }
    }
}
