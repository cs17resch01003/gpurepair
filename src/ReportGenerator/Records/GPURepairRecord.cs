namespace GPURepair.ReportGenerator.Records
{
    using CsvHelper.Configuration.Attributes;

    public class GPURepairRecord : GPURepairTimeRecord
    {
        [Name("blocks")]
        [Index(10)]
        public double? Blocks { get; set; }

        [Name("commands")]
        [Index(11)]
        public double? Commands { get; set; }

        [Name("call-commands")]
        [Index(12)]
        public double? CallCommands { get; set; }

        [Name("barriers")]
        [Index(13)]
        public double? Barriers { get; set; }

        [Name("grid-level-barriers")]
        [Index(14)]
        public double? GridLevelBarriers { get; set; }

        [Name("loop-barriers")]
        [Index(15)]
        public double? LoopBarriers { get; set; }

        [Name("changes")]
        [Index(16)]
        public double? Changes { get; set; }

        [Name("solution-calls")]
        [Index(17)]
        public double? SolutionCalls { get; set; }

        [Name("solution-grid")]
        [Index(18)]
        public double? SolutionGrid { get; set; }

        [Name("solution-loop")]
        [Index(19)]
        public double? SolutionLoop { get; set; }

        [Name("source-weight")]
        [Index(20)]
        public double? SourceWeight { get; set; }

        [Name("repaired-weight")]
        [Index(21)]
        public double? RepairedWeight { get; set; }

        [Name("runs-after-opt")]
        [Index(22)]
        public double? RunsAfterOpt { get; set; }

        [Name("fails-after-opt")]
        [Index(23)]
        public double? FailsAfterOpt { get; set; }

        [Name("mhs-count")]
        [Index(24)]
        public double? mhsCount { get; set; }

        [Name("mhs-time")]
        [Index(25)]
        public double? mhsTime { get; set; }

        [Name("maxsat-count")]
        [Index(26)]
        public double? MaxSATCount { get; set; }

        [Name("maxsat-time")]
        [Index(27)]
        public double? MaxSATTime { get; set; }

        [Name("sat-count")]
        [Index(28)]
        public double? SATCount { get; set; }

        [Name("sat-time")]
        [Index(29)]
        public double? SATTime { get; set; }

        [Name("ver-count")]
        [Index(30)]
        public double? VerCount { get; set; }

        [Name("ver-time")]
        [Index(31)]
        public double? VerTime { get; set; }

        [Name("opt-count")]
        [Index(32)]
        public double? OptCount { get; set; }

        [Name("opt-time")]
        [Index(33)]
        public double? OptTime { get; set; }

        [Name("solver-count")]
        [Index(34)]
        public double? SolverCount
        {
            get { return MaxSATCount + SATCount + OptCount; }
        }

        [Name("exception-message")]
        [Index(35)]
        public string ExceptionMessage { get; set; }
    }
}
