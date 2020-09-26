namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using GPURepair.ReportGenerator.Reports;

    public static class TexGenerator
    {
        public static void Generate(string directory)
        {
            IEnumerable<string> files = Directory.EnumerateFiles("template", "*", SearchOption.AllDirectories);
            foreach (string file in files)
            {
                string destination = file.Replace("template", directory + Path.DirectorySeparatorChar + "report");
                CopyFile(file, destination);
            }

            PrepareMain(directory);
        }

        private static void CopyFile(string source, string destination)
        {
            FileInfo info = new FileInfo(destination);
            if (!info.Directory.Exists)
                info.Directory.Create();

            File.Copy(source, destination);
        }

        private static void PrepareMain(string directory)
        {
            List<string> lines = new List<string>();
            if (FileParser.GPUVerify.Any())
            {
                lines.Add(@"\input{experiments/benchmark_summary}");
                PrepareBenchmarkSummary(directory);

                if (FileParser.GPURepair.Any())
                {
                    if (FileParser.Autosync.Any())
                    {
                        lines.Add(@"\input{experiments/tool_comparison}");
                        PrepareToolComparison(directory);
                    }

                    lines.Add(@"\input{experiments/source_information}");

                    if (FileParser.GPURepair_MaxSAT.Any() && FileParser.GPURepair_SAT.Any())
                        lines.Add(@"\input{experiments/solver_comparison}");
                    if (FileParser.GPURepair_Grid.Any() && FileParser.GPURepair_Inspection.Any() && FileParser.GPURepair_Grid_Inspection.Any())
                        lines.Add(@"\input{experiments/configuration_comparison}");
                }
            }

            string file = directory + Path.DirectorySeparatorChar + "report" + Path.DirectorySeparatorChar + "main.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@inputs@@", string.Join("\r\n", lines));
            File.WriteAllText(file, content);
        }

        private static void PrepareBenchmarkSummary(string directory)
        {
            int kernel_count = FileParser.GPUVerify.Count();
            int opencl_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("OpenCL")).Count();
            int cuda_count = kernel_count - opencl_count;

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "benchmark_summary.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", kernel_count.ToString());
            content = content.Replace("@@cuda_count@@", cuda_count.ToString());
            content = content.Replace("@@opencl_count@@", opencl_count.ToString());
            File.WriteAllText(file, content);

            int sdk_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("CUDASamples")).Count();
            int micro_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("MicroBenchmarks")).Count();
            int gpurepair_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("Bugged")).Count();
            int gpuverify_count = kernel_count - (sdk_count + micro_count + gpurepair_count);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "benchmark_summary.tex";

            content = File.ReadAllText(file);
            content = content.Replace("@@opencl_count@@", opencl_count.ToString());
            content = content.Replace("@@gpuverify_count@@", gpuverify_count.ToString());
            content = content.Replace("@@sdk_count@@", sdk_count.ToString());
            content = content.Replace("@@micro_count@@", micro_count.ToString());
            content = content.Replace("@@gpurepair_count@@", gpurepair_count.ToString());
            File.WriteAllText(file, content);
        }

        private static void PrepareToolComparison(string directory)
        {
            int autosync_repair_error = DataAnalyzer.ToolComparisonRecords.Where(x => x.AS_Status == "REPAIR_ERROR").Count();
            int autosync_gr_repaired = DataAnalyzer.ToolComparisonRecords.Where(x => x.AS_Status == "REPAIR_ERROR")
                .Where(x => x.GR_Status == "PASS").Count();
            int autosync_gr_timeout = DataAnalyzer.ToolComparisonRecords.Where(x => x.AS_Status == "REPAIR_ERROR")
                .Where(x => x.GR_Status == "FAIL(7)").Count();
            int autosync_gr_others = autosync_repair_error - (autosync_gr_repaired + autosync_gr_timeout);

            IEnumerable<ToolComparisonRecord> cuda =
                DataAnalyzer.ToolComparisonRecords.Where(x => !x.Kernel.Contains("OpenCL"));
            IEnumerable<ToolComparisonRecord> valid =
                cuda.Where(x => x.GV_Status == "PASS" || x.GV_Status == "FAIL(6)");

            int cuda_count = cuda.Count();
            int valid_count = valid.Count();
            int unrepairable_errors = cuda_count - valid_count;

            int autosync_faster = valid.Where(x => x.AS_Time < x.GR_Time).Count();
            int gpurepair_faster = valid_count - autosync_faster;
            int autosync_noeffort = valid.Where(x => x.GV_Status == "PASS").Count();
            int repairable_errors = valid_count - autosync_noeffort;
            int autosync_repairable_faster = valid.Where(x => x.GV_Status != "PASS" && x.AS_Time < x.GR_Time).Count();
            int gpurepair_repairable_faster = repairable_errors - autosync_repairable_faster;

            double autosync_time = cuda.Sum(x => x.AS_Time);
            double gpurepair_time = cuda.Sum(x => x.GR_Time);
            double autosync_median = Median(cuda.Select(x => x.AS_Time));
            double gpurepair_median = Median(cuda.Select(x => x.GR_Time));

            IEnumerable<ToolComparisonRecord> repaired =
                valid.Where(x => x.GV_Status == "FAIL(6)" && x.AS_Status == "REPAIRED" && x.GR_Status == "PASS");
            IEnumerable<ToolComparisonRecord> unchanged =
                valid.Where(x => x.GV_Status == "PASS" && x.AS_Status == "UNCHANGED" && x.GR_Status == "PASS" && x.GR_Changes == 0);

            int repaired_count = repaired.Count();
            int unchanged_count = unchanged.Count();
            int repaired_unchanged = repaired_count + unchanged_count;

            double autosync_both_vercalls = repaired.Union(unchanged).Sum(x => x.AS_VerCount);
            double gpurepair_both_vercalls = repaired.Union(unchanged).Sum(x => x.GR_VerCount);
            double autosync_repaired_vercalls = repaired.Sum(x => x.AS_VerCount);
            double gpurepair_repaired_vercalls = repaired.Sum(x => x.GR_VerCount);

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "tool_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@autosync_repair_error@@", autosync_repair_error.ToString());
            content = content.Replace("@@autosync_gr_repaired@@", autosync_gr_repaired.ToString());
            content = content.Replace("@@autosync_gr_timeout@@", autosync_gr_timeout.ToString());
            content = content.Replace("@@autosync_gr_others@@", autosync_gr_others.ToString());

            content = content.Replace("@@unrepairable_errors@@", unrepairable_errors.ToString());
            content = content.Replace("@@valid_count@@", valid_count.ToString());

            content = content.Replace("@@autosync_faster@@", autosync_faster.ToString());
            content = content.Replace("@@gpurepair_faster@@", gpurepair_faster.ToString());
            content = content.Replace("@@autosync_noeffort@@", autosync_noeffort.ToString());
            content = content.Replace("@@repairable_errors@@", repairable_errors.ToString());
            content = content.Replace("@@autosync_repairable_faster@@", autosync_repairable_faster.ToString());
            content = content.Replace("@@gpurepair_repairable_faster@@", gpurepair_repairable_faster.ToString());

            content = content.Replace("@@autosync_time@@", Convert.ToInt32(Math.Round(autosync_time)).ToString());
            content = content.Replace("@@gpurepair_time@@", Convert.ToInt32(Math.Round(gpurepair_time)).ToString());
            content = content.Replace("@@autosync_median@@", Math.Round(autosync_median, 2).ToString());
            content = content.Replace("@@gpurepair_median@@", Math.Round(gpurepair_median, 2).ToString());

            content = content.Replace("@@repaired_count@@", repaired_count.ToString());
            content = content.Replace("@@unchanged_count@@", unchanged_count.ToString());
            content = content.Replace("@@repaired_unchanged@@", repaired_unchanged.ToString());

            content = content.Replace("@@autosync_both_vercalls@@", autosync_both_vercalls.ToString());
            content = content.Replace("@@gpurepair_both_vercalls@@", gpurepair_both_vercalls.ToString());
            content = content.Replace("@@repaired_unchanged@@", autosync_repaired_vercalls.ToString());
            content = content.Replace("@@repaired_unchanged@@", gpurepair_repaired_vercalls.ToString());
            File.WriteAllText(file, content);

            PrepareResults(directory);
            PrepareToolTimeGraphs(directory);
            PrepareToolVerifierGraphs(directory);
        }

        private static void PrepareResults(string directory)
        {
            IEnumerable<ToolComparisonRecord> cuda =
                DataAnalyzer.ToolComparisonRecords.Where(x => !x.Kernel.Contains("OpenCL"));
            IEnumerable<ToolComparisonRecord> opencl =
                DataAnalyzer.ToolComparisonRecords.Where(x => x.Kernel.Contains("OpenCL"));

            IEnumerable<ToolComparisonRecord> cuda_verified = cuda.Where(x => x.GV_Status == "PASS");
            IEnumerable<ToolComparisonRecord> opencl_verified = opencl.Where(x => x.GV_Status == "PASS");

            IEnumerable<ToolComparisonRecord> cuda_verified_as_unchanged =
                cuda_verified.Where(x => x.AS_Status == "UNCHANGED");
            IEnumerable<ToolComparisonRecord> cuda_verified_gr_unchanged =
                cuda_verified.Where(x => x.GR_Status == "PASS" && x.GR_Changes == 0);
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_unchanged =
                opencl_verified.Where(x => x.GR_Status == "PASS" && x.GR_Changes == 0);

            IEnumerable<ToolComparisonRecord> cuda_verified_gr_changed =
                cuda_verified.Where(x => x.GR_Status == "PASS" && x.GR_Changes != 0);
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_changed =
                opencl_verified.Where(x => x.GR_Status == "PASS" && x.GR_Changes != 0);

            IEnumerable<ToolComparisonRecord> cuda_verified_as_timeout =
                cuda_verified.Where(x => x.AS_Status == "TIMEOUT");
            IEnumerable<ToolComparisonRecord> cuda_verified_gr_timeout =
                cuda_verified.Where(x => x.GR_Status == "FAIL(7)");
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_timeout =
                opencl_verified.Where(x => x.GR_Status == "FAIL(7)");

            int cuda_verified_as_errors = cuda_verified.Count() -
                (cuda_verified_as_unchanged.Count() + cuda_verified_as_timeout.Count());
            int cuda_verified_gr_errors = cuda_verified.Count() -
                (cuda_verified_gr_unchanged.Count() + cuda_verified_gr_changed.Count() + cuda_verified_gr_timeout.Count());
            int opencl_verified_gr_errors = opencl_verified.Count() -
                (opencl_verified_gr_unchanged.Count() + opencl_verified_gr_changed.Count() + opencl_verified_gr_timeout.Count());

            IEnumerable<ToolComparisonRecord> cuda_repairable = cuda.Where(x => x.GV_Status == "FAIL(6)");
            IEnumerable<ToolComparisonRecord> opencl_repairable = opencl.Where(x => x.GV_Status == "FAIL(6)");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_repaired =
                cuda_repairable.Where(x => x.AS_Status == "REPAIRED");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_repaired =
                cuda_repairable.Where(x => x.GR_Status == "PASS");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_repaired =
                opencl_repairable.Where(x => x.GR_Status == "PASS");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_repairerror =
                cuda_repairable.Where(x => x.AS_Status == "REPAIR_ERROR");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_repairerror =
                cuda_repairable.Where(x => x.GR_Status == "FAIL(202)");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_repairerror =
                opencl_repairable.Where(x => x.GR_Status == "FAIL(202)");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_timeout =
                cuda_repairable.Where(x => x.AS_Status == "TIMEOUT");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_timeout =
                cuda_repairable.Where(x => x.GR_Status == "FAIL(7)");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_timeout =
                opencl_repairable.Where(x => x.GR_Status == "FAIL(7)");

            int cuda_repairable_as_errors = cuda_repairable.Count() -
                (cuda_repairable_as_repaired.Count() + cuda_repairable_as_repairerror.Count() + cuda_repairable_as_timeout.Count());
            int cuda_repairable_gr_errors = cuda_repairable.Count() -
                (cuda_repairable_gr_repaired.Count() + cuda_repairable_gr_repairerror.Count() + cuda_repairable_gr_timeout.Count());
            int opencl_repairable_gr_errors = opencl_repairable.Count() -
                (opencl_repairable_gr_repaired.Count() + opencl_repairable_gr_repairerror.Count() + opencl_repairable_gr_timeout.Count());

            IEnumerable<ToolComparisonRecord> cuda_unrepairable =
                cuda.Where(x => x.GV_Status != "PASS" && x.GV_Status != "FAIL(6)");
            IEnumerable<ToolComparisonRecord> opencl_unrepairable =
                opencl.Where(x => x.GV_Status != "PASS" && x.GV_Status != "FAIL(6)");

            IEnumerable<ToolComparisonRecord> cuda_unrepairable_as_errors =
                cuda_unrepairable.Where(x => x.AS_Status == "ERROR");
            IEnumerable<ToolComparisonRecord> cuda_unrepairable_as_falsepositives =
                cuda_unrepairable.Where(x => x.AS_Status == "FALSE_POSITIVE");

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "results.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@cuda_verified@@", cuda_verified.Count().ToString());
            content = content.Replace("@@opencl_verified@@", opencl_verified.Count().ToString());

            content = content.Replace("@@cuda_verified_as_unchanged@@", cuda_verified_as_unchanged.Count().ToString());
            content = content.Replace("@@cuda_verified_gr_unchanged@@", cuda_verified_gr_unchanged.Count().ToString());
            content = content.Replace("@@opencl_verified_gr_unchanged@@", opencl_verified_gr_unchanged.Count().ToString());

            content = content.Replace("@@cuda_verified_gr_changed@@", cuda_verified_gr_changed.Count().ToString());
            content = content.Replace("@@opencl_verified_gr_changed@@", opencl_verified_gr_changed.Count().ToString());

            content = content.Replace("@@cuda_verified_as_timeout@@", cuda_verified_as_timeout.Count().ToString());
            content = content.Replace("@@cuda_verified_gr_timeout@@", cuda_verified_gr_timeout.Count().ToString());
            content = content.Replace("@@opencl_verified_gr_timeout@@", opencl_verified_gr_timeout.Count().ToString());

            content = content.Replace("@@cuda_verified_as_errors@@", cuda_verified_as_errors.ToString());
            content = content.Replace("@@cuda_verified_gr_errors@@", cuda_verified_gr_errors.ToString());
            content = content.Replace("@@opencl_verified_gr_errors@@", opencl_verified_gr_errors.ToString());

            content = content.Replace("@@cuda_repairable@@", cuda_repairable.Count().ToString());
            content = content.Replace("@@opencl_repairable@@", opencl_repairable.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_repaired@@", cuda_repairable_as_repaired.Count().ToString());
            content = content.Replace("@@cuda_repairable_gr_repaired@@", cuda_repairable_gr_repaired.Count().ToString());
            content = content.Replace("@@opencl_repairable_gr_repaired@@", opencl_repairable_gr_repaired.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_repairerror@@", cuda_repairable_as_repaired.Count().ToString());
            content = content.Replace("@@cuda_repairable_gr_repairerror@@", cuda_repairable_gr_repaired.Count().ToString());
            content = content.Replace("@@opencl_repairable_gr_repairerror@@", opencl_repairable_gr_repaired.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_timeout@@", cuda_repairable_as_timeout.Count().ToString());
            content = content.Replace("@@cuda_repairable_gr_timeout@@", cuda_repairable_gr_timeout.Count().ToString());
            content = content.Replace("@@opencl_repairable_gr_timeout@@", opencl_repairable_gr_timeout.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_errors@@", cuda_repairable_as_errors.ToString());
            content = content.Replace("@@cuda_repairable_gr_errors@@", cuda_repairable_gr_errors.ToString());
            content = content.Replace("@@opencl_repairable_gr_errors@@", opencl_repairable_gr_errors.ToString());

            content = content.Replace("@@cuda_unrepairable@@", cuda_unrepairable.Count().ToString());
            content = content.Replace("@@opencl_unrepairable@@", opencl_unrepairable.Count().ToString());

            content = content.Replace("@@cuda_unrepairable_as_errors@@", cuda_unrepairable_as_errors.ToString());
            content = content.Replace("@@cuda_unrepairable_as_falsepositives@@", cuda_unrepairable_as_falsepositives.ToString());
            File.WriteAllText(file, content);
        }

        private static void PrepareToolTimeGraphs(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_repaired.dat";

            IEnumerable<string> repaired = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AS_Status == "REPAIRED" && x.GR_Status == "PASS")
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GR_Time, 2).ToString(),
                    Math.Round(x.AS_Time, 2).ToString(),
                    "a"
                }));

            IEnumerable<string> unchanged = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AS_Status == "UNCHANGED" && x.GR_Status == "PASS" && x.GR_Changes == 0)
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GR_Time, 2).ToString(),
                    Math.Round(x.AS_Time, 2).ToString(),
                    "a"
                }));

            string content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", repaired.Union(unchanged)));
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_scatter.dat";

            IEnumerable<string> data = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AS_Time > 0 && x.GR_Time > 0)
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GR_Time, 2).ToString(),
                    Math.Round(x.AS_Time, 2).ToString(),
                    "a"
                }));

            content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);
        }

        private static void PrepareToolVerifierGraphs(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "verifier_calls_all.dat";

            IEnumerable<ToolComparisonRecord> repaired = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AS_Status == "REPAIRED" && x.GR_Status == "PASS")
                .Where(x => x.AS_VerCount > 0 && x.GR_VerCount > 0);

            IEnumerable<ToolComparisonRecord> unchanged = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AS_Status == "UNCHANGED" && x.GR_Status == "PASS" && x.GR_Changes == 0)
                .Where(x => x.AS_VerCount > 0 && x.GR_VerCount > 0);

            List<string> data = new List<string>();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                {
                    // calls more than 6 are considered as 6
                    int count = repaired.Union(unchanged).Count(x =>
                        ((x.GR_VerCount > 6 ? 6 : x.GR_VerCount) == (i + 1)) &&
                        ((x.AS_VerCount > 6 ? 6 : x.AS_VerCount) == (j + 1))
                    );

                    if (count != 0)
                        data.Add(string.Join("\t", new string[] { (i + 1).ToString(), (j + 1).ToString(),
                            (count > 5 ? 5 + (count / 10) : count).ToString(), count.ToString() }));
                }

            string content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "verifier_calls_fixes.dat";

            data = new List<string>();
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                {
                    // calls more than 6 are considered as 6
                    int count = repaired.Count(x =>
                        ((x.GR_VerCount > 6 ? 6 : x.GR_VerCount) == (i + 1)) &&
                        ((x.AS_VerCount > 6 ? 6 : x.AS_VerCount) == (j + 1))
                    );

                    if (count != 0)
                        data.Add(string.Join("\t", new string[] { (i + 1).ToString(), (j + 1).ToString(),
                            (count > 5 ? 5 + (count / 10) : count).ToString(), count.ToString() }));
                }

            content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);
        }

        private static double Median(IEnumerable<double> values)
        {
            int count = values.Count();
            int index = count / 2;

            IEnumerable<double> sorted = values.OrderBy(n => n);
            double median = count % 2 != 0 ? values.ElementAt(index) :
                (values.ElementAt(index) + values.ElementAt(index - 1)) / 2;

            return median;
        }
    }
}
