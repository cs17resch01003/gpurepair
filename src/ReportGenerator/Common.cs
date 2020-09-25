namespace GPURepair.ReportGenerator
{
    using System.Collections.Generic;

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
    }
}
