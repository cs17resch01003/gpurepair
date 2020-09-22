namespace ReportGenerator
{
    using CsvHelper.Configuration.Attributes;

    public class Metrics
    {
        [Name("kernel")]
        public string Kernel { get; set; }

        [Name("blocks")]
        public int Blocks { get; set; }

        [Name("commands")]
        public int Commands { get; set; }

        [Name("call commands")]
        public int CallCommands { get; set; }

        [Name("barriers")]
        public int Barriers { get; set; }

        [Name("grid-level barriers")]
        public int GridLevelBarriers { get; set; }

        [Name("loop barriers")]
        public int BarriersInsideLoop { get; set; }

        [Name("changes")]
        public int Changes { get; set; }

        [Name("solution (calls)")]
        public int Solution_BarriersBetweenCalls { get; set; }

        [Name("solution (grid)")]
        public int Solution_GridLevelBarriers { get; set; }

        [Name("solution (loop)")]
        public int Solution_BarriersInsideLoop { get; set; }

        [Name("source weight")]
        public int SourceWeight { get; set; }

        [Name("repaired weight")]
        public int RepairedWeight { get; set; }

        [Name("v-runs after opt")]
        public int VerifierRunsAfterOptimization { get; set; }

        [Name("v-fail after opt")]
        public int VerifierFailuresAfterOptimization { get; set; }

        [Name("mhs count")]
        public int MHS_Count { get; set; }

        [Name("mhs time")]
        public int MHS_Time { get; set; }

        [Name("maxsat count")]
        public int MaxSAT_Count { get; set; }

        [Name("maxsat time")]
        public int MaxSAT_Time { get; set; }

        [Name("sat count")]
        public int SAT_Count { get; set; }

        [Name("sat time")]
        public int SAT_Time { get; set; }

        [Name("ver count")]
        public int Verification_Count { get; set; }

        [Name("ver time")]
        public int Verification_Time { get; set; }

        [Name("opt count")]
        public int Optimization_Count { get; set; }

        [Name("opt time")]
        public int Optimization_Time { get; set; }

        [Name("solver count")]
        public int Total_Solver_Count
        {
            get { return MaxSAT_Count + SAT_Count + Optimization_Count; }
        }

        [Name("exception message")]
        public string ExceptionMessage { get; set; }
    }
}
