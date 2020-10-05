namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.ReportGenerator.Records;
    using GPURepair.ReportGenerator.Reports;

    public static class Common
    {
        public static string StandardizeKernelName(string kernel)
        {
            kernel = kernel.Replace("kernel.cu", string.Empty).Replace("kernel.cl", string.Empty).Replace("metrics.log", string.Empty);
            List<string> categories = new List<string>()
            {
                "Bugged",
                "Bugged_Cooperative",
                "CUDA",
                "CUDASamples",
                "MicroBenchmarks",
                "OpenCL"
            };

            kernel = kernel.Replace('\\', '/');
            string[] parts = kernel.Split(new char[] { '/' });

            List<string> temp = new List<string>();
            for (int i = parts.Length - 1; i >= 0; i--)
            {
                if (string.IsNullOrWhiteSpace(parts[i]))
                    continue;

                temp.Add(parts[i]);
                if (categories.Contains(parts[i]))
                    break;
            }

            temp.Reverse();
            return string.Join("/", temp.ToArray());
        }

        public static string Result(this GPURepairRecord record, BaseReportRecord report_record)
        {
            if (report_record.GPUVerify.Result != "PASS")
                return record.Result;

            if (record.ResultEnum == GPURepairTimeRecord.Status.Success && record.Changes == 0)
                return "UNCHANGED";
            return record.Result;
        }

        public static string Replace(this string input, string oldValue, int newValue)
        {
            return input.Replace(oldValue, newValue.ToString());
        }

        public static string Replace(this string input, string oldValue, double newValue, int? precision = null)
        {
            if (precision == null)
                return input.Replace(oldValue, Convert.ToInt32(Math.Round(newValue)));
            return input.Replace(oldValue, Math.Round(newValue, precision.Value).ToString("0.00"));
        }

        public static string Replace<T>(this string input, string oldValue, IEnumerable<T> newValue)
        {
            return input.Replace(oldValue, newValue.Count());
        }
    }
}
