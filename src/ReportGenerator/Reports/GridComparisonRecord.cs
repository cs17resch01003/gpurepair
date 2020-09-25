namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;

    public class GridComparisonRecord
    {
        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("gv-status")]
        [Index(1)]
        public string GV_Status { get; set; }

        [Name("en-status")]
        [Index(2)]
        public string Enabled_Status { get; set; }

        [Name("en-time")]
        [Index(3)]
        public double Enabled_Time { get; set; }

        [Name("en-solver-count")]
        [Index(4)]
        public double Enabled_SolverCount { get; set; }

        [Name("en-ver-count")]
        [Index(5)]
        public double Enabled_VerCount { get; set; }

        [Name("dis-status")]
        [Index(6)]
        public string Disabled_Status { get; set; }

        [Name("dis-time")]
        [Index(7)]
        public double Disabled_Time { get; set; }

        [Name("dis-solver-count")]
        [Index(8)]
        public double Disabled_SolverCount { get; set; }

        [Name("dis-ver-count")]
        [Index(9)]
        public double Disabled_VerCount { get; set; }
    }
}
