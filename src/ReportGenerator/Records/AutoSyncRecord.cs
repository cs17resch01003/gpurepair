namespace GPURepair.ReportGenerator.Records
{
    using System.Collections.Generic;
    using CsvHelper.Configuration.Attributes;

    public class AutoSyncRecord
    {
        [Index(0)]
        public string Kernel { get; set; }

        [Index(1)]
        public double? Time { get; set; }

        [Index(2)]
        public double? VerCount { get; set; }

        [Index(3)]
        public double? Changes { get; set; }

        [Index(4)]
        public string Exception { get; set; }

        public Status Result
        {
            get
            {
                List<string> falsePositives = new List<string>()
                {
                    "CUDA/atomics/test_abstraction_enforced",
                    "CUDA/function_pointers/basic_argument_fail",
                    "CUDA/function_pointers/basic_assignment_fail",
                    "CUDA/function_pointers/constant_value",
                    "CUDA/function_pointers/funcptr_to_ptr_add",
                    "CUDA/function_pointers/pass_struct/call",
                    "CUDA/function_pointers/soundness_issue",
                    "CUDA/memcpy/null_dst",
                    "CUDA/memcpy/null_src",
                    "CUDA/memcpy/unhandled_varlen",
                    "CUDA/memset/null_dst",
                    "CUDA/memset/unhandled_varlen",
                    "CUDA/memset/unhandled_varval",
                    "CUDA/misc/fail/miscfail1",
                    "CUDA/misc/fail/miscfail2",
                    "CUDA/misc/fail/miscfail5",
                    "CUDA/misc/fail/miscfail7",
                    "CUDA/misc/fail/miscfail8",
                    "CUDA/misc/pass/misc3",
                    "CUDA/param_values/value_in_assert",
                    "CUDA/return_val/char",
                    "CUDA/return_val/longlong",
                    "CUDA/warpsync/equality_abstraction_issue",
                    "CUDASamples/3_Imaging_stereoDisparity_stereoDisparity"
                };

                if (falsePositives.Contains(Kernel))
                    return Status.FalsePositive;
                else if (Exception == "Timeout!")
                    return Status.Timeout;
                else if (VerCount == -1 && Changes == -1)
                    return Status.Error;
                else if (Changes == -1)
                    return Status.RepairError;
                else if (Changes > 0)
                    return Status.Repaired;
                else
                    return Status.Unchanged;
            }
        }

        public class Status
        {
            public static Status Unchanged = new Status("UNCHANGED", 4);

            public static Status Repaired = new Status("REPAIRED", 5);

            public static Status RepairError = new Status("REPAIR_ERROR", 3);

            public static Status Error = new Status("ERROR", 1);

            public static Status Timeout = new Status("TIMEOUT", 2);

            public static Status FalsePositive = new Status("FALSE_POSITIVE", 6);

            private Status(string value, int priority)
            {
                Value = value;
                Priority = priority;
            }

            public string Value { get; set; }

            public int Priority { get; set; }

            public override string ToString()
            {
                return Value;
            }
        }
    }
}
