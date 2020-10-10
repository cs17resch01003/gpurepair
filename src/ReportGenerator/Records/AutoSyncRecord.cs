namespace GPURepair.ReportGenerator.Records
{
    using CsvHelper.Configuration.Attributes;

    public class AutoSyncRecord
    {
        private Status result = null;

        [Index(0)]
        public string Kernel { get; set; }

        [Index(1)]
        public double Time { get; set; }

        [Index(2)]
        public double VerCount { get; set; }

        [Index(3)]
        public double Changes { get; set; }

        [Index(4)]
        public string Exception { get; set; }

        [Ignore]
        public Status Result
        {
            get
            {
                if (result != null && !string.IsNullOrWhiteSpace(result.Value))
                    return result;

                if (Exception == "Timeout!")
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

            set { result = value; }
        }

        public class Status
        {
            public static Status Unchanged = new Status("UNCHANGED", 4);

            public static Status Repaired = new Status("REPAIRED", 5);

            public static Status RepairError = new Status("REPAIR_ERROR", 3);

            public static Status Error = new Status("ERROR", 1);

            public static Status Timeout = new Status("TIMEOUT", 2);

            public static Status FalsePositive = new Status("FALSE_POSITIVE", 6);

            public Status()
            {
                Value = null;
                Priority = null;
            }

            private Status(string value, int priority)
            {
                Value = value;
                Priority = priority;
            }

            public string Value { get; set; }

            public int? Priority { get; set; }

            public override string ToString()
            {
                return Value;
            }
        }
    }
}
