namespace ReportGenerator
{
    using CsvHelper.Configuration.Attributes;

    public class Metrics
    {
        [Name("kernel")]
        [Index(1)]
        public string Kernel { get; set; }

        [Name("blocks")]
        [Index(2)]
        public int Blocks { get; set; }

        [Name("commands")]
        [Index(3)]
        public int Commands { get; set; }

        [Name("call-commands")]
        [Index(4)]
        public int CallCommands { get; set; }

        [Name("barriers")]
        [Index(5)]
        public int Barriers { get; set; }

        [Name("grid-level-barriers")]
        [Index(6)]
        public int GridLevelBarriers { get; set; }

        [Name("loop-barriers")]
        [Index(7)]
        public int LoopBarriers { get; set; }

        [Name("changes")]
        [Index(8)]
        public int Changes { get; set; }

        [Name("solution-calls")]
        [Index(9)]
        public int SolutionCalls { get; set; }

        [Name("solution-grid")]
        [Index(10)]
        public int SolutionGrid { get; set; }

        [Name("solution-loop")]
        [Index(11)]
        public int SolutionLoop { get; set; }

        [Name("source-weight")]
        [Index(12)]
        public int SourceWeight { get; set; }

        [Name("repaired-weight")]
        [Index(13)]
        public int RepairedWeight { get; set; }

        [Name("runs-after-opt")]
        [Index(14)]
        public int RunsAfterOpt { get; set; }

        [Name("fails-after-opt")]
        [Index(15)]
        public int FailsAfterOpt { get; set; }

        [Name("mhs-count")]
        [Index(16)]
        public int mhsCount { get; set; }

        [Name("mhs-time")]
        [Index(17)]
        public int mhsTime { get; set; }

        [Name("maxsat-count")]
        [Index(18)]
        public int MaxSATCount { get; set; }

        [Name("maxsat-time")]
        [Index(19)]
        public int MaxSATTime { get; set; }

        [Name("sat-count")]
        [Index(20)]
        public int SATCount { get; set; }

        [Name("sat-time")]
        [Index(21)]
        public int SATTime { get; set; }

        [Name("ver-count")]
        [Index(22)]
        public int VerCount { get; set; }

        [Name("ver-time")]
        [Index(23)]
        public int VerTime { get; set; }

        [Name("opt-count")]
        [Index(24)]
        public int OptCount { get; set; }

        [Name("opt-time")]
        [Index(25)]
        public int OptTime { get; set; }

        [Name("solver-count")]
        [Index(26)]
        public int SolverCount
        {
            get { return MaxSATCount + SATCount + OptCount; }
        }

        [Name("exception-message")]
        [Index(27)]
        public string ExceptionMessage { get; set; }
    }
}
