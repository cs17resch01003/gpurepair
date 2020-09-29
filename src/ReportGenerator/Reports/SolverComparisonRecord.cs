﻿namespace GPURepair.ReportGenerator.Reports
{
    using CsvHelper.Configuration.Attributes;

    public class SolverComparisonRecord
    {
        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("gv-status")]
        [Index(1)]
        public string GV_Status { get; set; }

        [Name("mhs-status")]
        [Index(2)]
        public string mhs_Status { get; set; }

        [Name("mhs-time")]
        [Index(3)]
        public double mhs_Time { get; set; }

        [Name("mhs-solver-count")]
        [Index(4)]
        public double mhs_SolverCount { get; set; }

        [Name("mhs-ver-count")]
        [Index(5)]
        public double mhs_VerCount { get; set; }

        [Name("maxsat-status")]
        [Index(6)]
        public string MaxSAT_Status { get; set; }

        [Name("maxsat-time")]
        [Index(7)]
        public double MaxSAT_Time { get; set; }

        [Name("maxsat-solver-count")]
        [Index(8)]
        public double MaxSAT_SolverCount { get; set; }

        [Name("maxsat-ver-count")]
        [Index(9)]
        public double MaxSAT_VerCount { get; set; }

        [Name("sat-status")]
        [Index(10)]
        public string SAT_Status { get; set; }

        [Name("sat-time")]
        [Index(11)]
        public double SAT_Time { get; set; }

        [Name("sat-solver-count")]
        [Index(12)]
        public double SAT_SolverCount { get; set; }

        [Name("sat-ver-count")]
        [Index(13)]
        public double SAT_VerCount { get; set; }
    }
}