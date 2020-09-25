namespace GPURepair.ReportGenerator.Records
{
    using System.Collections.Generic;
    using System.IO;
    using CsvHelper.Configuration.Attributes;

    public class GPUVerifyRecord
    {
        private static List<string> assertionErrors;

        private static List<string> invariantErrors;

        static GPUVerifyRecord()
        {
            assertionErrors = CsvWrapper.ReadLines(@"manual\assertionerrors.csv");
            invariantErrors = CsvWrapper.ReadLines(@"manual\invarianterrors.csv");
        }

        public Status ResultEnum;

        [Name("kernel")]
        [Index(0)]
        public string Kernel { get; set; }

        [Name("status")]
        [Index(1)]
        public string Result
        {
            get
            {
                if (assertionErrors.Contains(Kernel))
                    return "FAIL(6)-ASSERT";
                else if (invariantErrors.Contains(Kernel))
                    return "FAIL(6)-INV";
                else
                    return ResultEnum.Value;
            }
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

        [Name("vcgen")]
        [Index(5)]
        public double VCGen { get; set; }

        [Name("cruncher")]
        [Index(6)]
        public double Cruncher { get; set; }

        [Name("boogiedriver")]
        [Index(7)]
        public double BoogieDriver { get; set; }

        [Name("total")]
        [Index(8)]
        public double Total { get; set; }

        public class Status
        {
            public static Status Success = new Status("PASS", 7);

            public static Status CommandLineError = new Status("FAIL(1)", 1);

            public static Status ClangError = new Status("FAIL(2)", 2);

            public static Status OptError = new Status("FAIL(3)", 3);

            public static Status BugleError = new Status("FAIL(4)", 4);

            public static Status VCGenError = new Status("FAIL(5)", 5);

            public static Status VerifierError = new Status("FAIL(6)", 6);

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

                throw new InvalidDataException();
            }

            public override string ToString()
            {
                return Value;
            }
        }
    }
}
