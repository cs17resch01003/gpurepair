namespace GPURepair.ReportGenerator.Records
{
    using CsvHelper.Configuration.Attributes;

    public class GPURepairRecord : GPURepairTimeRecord
    {
        [Name("blocks")]
        [Index(10)]
        public int Blocks { get; set; }

        [Name("commands")]
        [Index(11)]
        public int Commands { get; set; }

        [Name("call-commands")]
        [Index(12)]
        public int CallCommands { get; set; }

        [Name("barriers")]
        [Index(13)]
        public int Barriers { get; set; }

        [Name("grid-level-barriers")]
        [Index(14)]
        public int GridLevelBarriers { get; set; }

        [Name("loop-barriers")]
        [Index(15)]
        public int LoopBarriers { get; set; }

        [Name("changes")]
        [Index(16)]
        public int Changes { get; set; }

        [Name("solution-calls")]
        [Index(17)]
        public int SolutionCalls { get; set; }

        [Name("solution-grid")]
        [Index(18)]
        public int SolutionGrid { get; set; }

        [Name("solution-loop")]
        [Index(19)]
        public int SolutionLoop { get; set; }

        [Name("source-weight")]
        [Index(20)]
        public int SourceWeight { get; set; }

        [Name("repaired-weight")]
        [Index(21)]
        public int RepairedWeight { get; set; }

        [Name("runs-after-opt")]
        [Index(22)]
        public int RunsAfterOpt { get; set; }

        [Name("fails-after-opt")]
        [Index(23)]
        public int FailsAfterOpt { get; set; }

        [Name("mhs-count")]
        [Index(24)]
        public int mhsCount { get; set; }

        [Name("mhs-time")]
        [Index(25)]
        public int mhsTime { get; set; }

        [Name("maxsat-count")]
        [Index(26)]
        public int MaxSATCount { get; set; }

        [Name("maxsat-time")]
        [Index(27)]
        public int MaxSATTime { get; set; }

        [Name("sat-count")]
        [Index(28)]
        public int SATCount { get; set; }

        [Name("sat-time")]
        [Index(29)]
        public int SATTime { get; set; }

        [Name("ver-count")]
        [Index(30)]
        public int VerCount { get; set; }

        [Name("ver-time")]
        [Index(31)]
        public int VerTime { get; set; }

        [Name("opt-count")]
        [Index(32)]
        public int OptCount { get; set; }

        [Name("opt-time")]
        [Index(33)]
        public int OptTime { get; set; }

        [Name("solver-count")]
        [Index(34)]
        public int SolverCount
        {
            get { return MaxSATCount + SATCount + OptCount; }
        }

        [Name("exception-message")]
        [Index(35)]
        public string ExceptionMessage { get; set; }
    }
}
