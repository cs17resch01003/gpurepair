namespace GPURepair.ReportGenerator.Records
{
    using CsvHelper.Configuration.Attributes;

    public class GPURepairRecord : GPURepairTimeRecord
    {
        [Name("lines")]
        [Index(10)]
        public double Lines { get; set; }

        [Name("blocks")]
        [Index(11)]
        public double Blocks { get; set; }

        [Name("commands")]
        [Index(12)]
        public double Commands { get; set; }

        [Name("call-commands")]
        [Index(13)]
        public double CallCommands { get; set; }

        [Name("barriers")]
        [Index(14)]
        public double Barriers { get; set; }

        [Name("grid-level-barriers")]
        [Index(15)]
        public double GridLevelBarriers { get; set; }

        [Name("loop-barriers")]
        [Index(16)]
        public double LoopBarriers { get; set; }

        [Name("changes")]
        [Index(17)]
        public double Changes { get; set; }

        [Name("solution-calls")]
        [Index(18)]
        public double SolutionCalls { get; set; }

        [Name("solution-grid")]
        [Index(19)]
        public double SolutionGrid { get; set; }

        [Name("solution-loop")]
        [Index(20)]
        public double SolutionLoop { get; set; }

        [Name("source-weight")]
        [Index(21)]
        public double SourceWeight { get; set; }

        [Name("repaired-weight")]
        [Index(22)]
        public double RepairedWeight { get; set; }

        [Name("runs-after-opt")]
        [Index(23)]
        public double RunsAfterOpt { get; set; }

        [Name("fails-after-opt")]
        [Index(24)]
        public double FailsAfterOpt { get; set; }

        [Name("mhs-count")]
        [Index(25)]
        public double mhsCount { get; set; }

        [Name("mhs-time")]
        [Index(26)]
        public double mhsTime { get; set; }

        [Name("maxsat-count")]
        [Index(27)]
        public double MaxSATCount { get; set; }

        [Name("maxsat-time")]
        [Index(28)]
        public double MaxSATTime { get; set; }

        [Name("sat-count")]
        [Index(29)]
        public double SATCount { get; set; }

        [Name("sat-time")]
        [Index(30)]
        public double SATTime { get; set; }

        [Name("ver-count")]
        [Index(31)]
        public double VerCount { get; set; }

        [Name("ver-time")]
        [Index(32)]
        public double VerTime { get; set; }

        [Name("opt-count")]
        [Index(33)]
        public double OptCount { get; set; }

        [Name("opt-time")]
        [Index(34)]
        public double OptTime { get; set; }

        [Name("solver-count")]
        [Index(35)]
        public double SolverCount
        {
            get { return MaxSATCount + SATCount + OptCount; }
        }

        [Name("exception-message")]
        [Index(36)]
        public string ExceptionMessage { get; set; }
    }
}
