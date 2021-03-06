﻿namespace GPURepair.ReportGenerator.Records
{
    using System.IO;
    using CsvHelper.Configuration.Attributes;

    public class GPURepairTimeRecord
    {
        private double total;

        public Status ResultEnum;

        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("status")]
        [Index(1)]
        public string Result
        {
            get { return ResultEnum.Value; }
            set { ResultEnum = Status.GetStatus(value); }
        }

        [Name("clang")]
        [Index(2)]
        public double Clang { get; set; }

        [Name("opt")]
        [Index(3)]
        public double Opt { get; set; }

        [Name("bugle")]
        [Index(4)]
        public double Bugle { get; set; }

        [Name("instrumentation")]
        [Index(5)]
        public double Instrumentation { get; set; }

        [Name("vcgen")]
        [Index(6)]
        public double VCGen { get; set; }

        [Name("cruncher")]
        [Index(7)]
        public double Cruncher { get; set; }

        [Name("repair")]
        [Index(8)]
        public double Repair { get; set; }

        [Name("total")]
        [Index(9)]
        public double Total
        {
            get { return total > 300 ? 300 : total; }
            set { total = value; }
        }

        public class Status
        {
            public static Status Success = new Status("PASS", 10);

            public static Status CommandLineError = new Status("FAIL(1)", 2);

            public static Status ClangError = new Status("FAIL(2)", 3);

            public static Status OptError = new Status("FAIL(3)", 4);

            public static Status BugleError = new Status("FAIL(4)", 5);

            public static Status VCGenError = new Status("FAIL(5)", 6);

            public static Status VerifierError = new Status("FAIL(6)", 7);

            public static Status Timeout = new Status("FAIL(7)", 1);

            public static Status AssertionError = new Status("FAIL(201)", 8);

            public static Status RepairError = new Status("FAIL(202)", 9);

            private Status(string value, int priority)
            {
                Value = value;
                Priority = priority;
            }

            public string Value { get; set; }

            public int Priority { get; set; }

            public static Status GetStatus(string status)
            {
                if (Success.Value == status) return Success;
                if (CommandLineError.Value == status) return CommandLineError;
                if (ClangError.Value == status) return ClangError;
                if (OptError.Value == status) return OptError;
                if (BugleError.Value == status) return BugleError;
                if (VCGenError.Value == status) return VCGenError;
                if (VerifierError.Value == status) return VerifierError;
                if (Timeout.Value == status) return Timeout;
                if (AssertionError.Value == status) return AssertionError;
                if (RepairError.Value == status) return RepairError;

                throw new InvalidDataException();
            }

            public override string ToString()
            {
                return Value;
            }
        }
    }
}
