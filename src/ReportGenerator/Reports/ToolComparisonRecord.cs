namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;

    public class ToolComparisonRecord
    {
        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("gv-status")]
        [Index(1)]
        public string GV_Status { get; set; }

        [Name("as-status")]
        [Index(2)]
        public string AS_Status { get; set; }

        [Name("as-time")]
        [Index(3)]
        public double AS_Time { get; set; }

        [Name("as-ver-count")]
        [Index(4)]
        public double AS_VerCount { get; set; }

        [Name("as-changes")]
        [Index(5)]
        public double AS_Changes { get; set; }

        [Name("gr-status")]
        [Index(6)]
        public string GR_Status { get; set; }

        [Name("gr-time")]
        [Index(7)]
        public double GR_Time { get; set; }

        [Name("gr-ver-count")]
        [Index(8)]
        public double GR_VerCount { get; set; }

        [Name("gr-changes")]
        [Index(9)]
        public double GR_Changes { get; set; }
    }
}
