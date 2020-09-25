namespace GPURepair.ReportGenerator.Records
{
    using CsvHelper.Configuration.Attributes;

    public class AutoSyncOutRecord
    {
        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("status")]
        [Index(1)]
        public string Status { get; set; }

        [Name("time")]
        [Index(2)]
        public double? Time { get; set; }

        [Name("ver-count")]
        [Index(3)]
        public double? VerCount { get; set; }

        [Name("changes")]
        [Index(4)]
        public double? Changes { get; set; }

        [Name("exception")]
        [Index(5)]
        public string Exception { get; set; }
    }
}
