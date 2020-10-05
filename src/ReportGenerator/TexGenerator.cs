namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using GPURepair.ReportGenerator.Records;
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
                    PrepareSourceInformation(directory);

                    if (FileParser.GPURepair_MaxSAT.Any() && FileParser.GPURepair_SAT.Any())
                    {
                        lines.Add(@"\input{experiments/solver_comparison}");
                        PrepareSolverComparison(directory);
                    }

                    if (FileParser.GPURepair_Grid.Any() && FileParser.GPURepair_Inspection.Any() && FileParser.GPURepair_Grid_Inspection.Any())
                    {
                        lines.Add(@"\input{experiments/configuration_comparison}");
                        PrepareConfigurationComparison(directory);
                    }
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
            int cooperative_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("Bugged_Cooperative")).Count();
            int gpurepair_count = FileParser.GPUVerify.Where(x => x.Kernel.Contains("Bugged")).Count();
            int gpuverify_count = kernel_count - (sdk_count + micro_count + gpurepair_count);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "benchmark_summary.tex";

            content = File.ReadAllText(file);
            content = content.Replace("@@opencl_count@@", opencl_count);
            content = content.Replace("@@gpuverify_count@@", gpuverify_count);
            content = content.Replace("@@sdk_count@@", sdk_count);
            content = content.Replace("@@micro_count@@", micro_count);
            content = content.Replace("@@cooperative_count@@", cooperative_count);
            content = content.Replace("@@gpurepair_count@@", gpurepair_count);
            File.WriteAllText(file, content);
        }

        private static void PrepareToolComparison(string directory)
        {
            int autosync_repair_error =
                DataAnalyzer.ToolComparisonRecords.Count(x => x.AutoSync_Status == "REPAIR_ERROR");
            int autosync_gr_repaired =
                DataAnalyzer.ToolComparisonRecords.Where(x => x.AutoSync_Status == "REPAIR_ERROR")
                .Count(x => x.GPURepair_Status == "PASS");
            int autosync_gr_timeout =
                DataAnalyzer.ToolComparisonRecords.Where(x => x.AutoSync_Status == "REPAIR_ERROR")
                .Count(x => x.GPURepair_Status == "FAIL(7)");
            int autosync_gr_others = autosync_repair_error - (autosync_gr_repaired + autosync_gr_timeout);

            IEnumerable<ToolComparisonRecord> cuda =
                DataAnalyzer.ToolComparisonRecords.Where(x => !x.Kernel.Contains("OpenCL"));
            IEnumerable<ToolComparisonRecord> valid =
                cuda.Where(x => x.GPUVerify_Status == "PASS" || x.GPUVerify_Status == "FAIL(6)");

            int cuda_count = cuda.Count();
            int valid_count = valid.Count();
            int unrepairable_errors = cuda_count - valid_count;

            int autosync_faster = valid.Count(x => x.AutoSync_Time < x.GPURepair_Time);
            int gpurepair_faster = valid_count - autosync_faster;
            int autosync_noeffort = valid.Count(x => x.GPUVerify_Status == "PASS");
            int repairable_errors = valid_count - autosync_noeffort;
            int autosync_repairable_faster =
                valid.Count(x => x.GPUVerify_Status != "PASS" && x.AutoSync_Time < x.GPURepair_Time);
            int gpurepair_repairable_faster = repairable_errors - autosync_repairable_faster;

            double autosync_time = valid.Sum(x => x.AutoSync_Time);
            double gpurepair_time = valid.Sum(x => x.GPURepair_Time);
            double autosync_median = Median(valid.Select(x => x.AutoSync_Time));
            double gpurepair_median = Median(valid.Select(x => x.GPURepair_Time));

            IEnumerable<ToolComparisonRecord> repaired = valid.Where(x => x.GPUVerify_Status == "FAIL(6)")
                .Where(x => x.AutoSync_Status == "REPAIRED" && x.GPURepair_Status == "PASS");
            IEnumerable<ToolComparisonRecord> unchanged = valid.Where(x => x.GPUVerify_Status == "PASS")
                .Where(x => x.AutoSync_Status == "UNCHANGED" && x.GPURepair_Status == "UNCHANGED");

            int repaired_count = repaired.Count();
            int unchanged_count = unchanged.Count();
            int repaired_unchanged = repaired_count + unchanged_count;

            double autosync_both_vercalls = repaired.Union(unchanged).Sum(x => x.AutoSync_VerCount);
            double gpurepair_both_vercalls = repaired.Union(unchanged).Sum(x => x.GPURepair_VerCount);
            double autosync_repaired_vercalls = repaired.Sum(x => x.AutoSync_VerCount);
            double gpurepair_repaired_vercalls = repaired.Sum(x => x.GPURepair_VerCount);

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "tool_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@autosync_repair_error@@", autosync_repair_error);
            content = content.Replace("@@autosync_gr_repaired@@", autosync_gr_repaired);
            content = content.Replace("@@autosync_gr_timeout@@", autosync_gr_timeout);
            content = content.Replace("@@autosync_gr_others@@", autosync_gr_others);

            content = content.Replace("@@unrepairable_errors@@", unrepairable_errors);
            content = content.Replace("@@valid_count@@", valid_count);

            content = content.Replace("@@autosync_faster@@", autosync_faster);
            content = content.Replace("@@gpurepair_faster@@", gpurepair_faster);
            content = content.Replace("@@autosync_noeffort@@", autosync_noeffort);
            content = content.Replace("@@repairable_errors@@", repairable_errors);
            content = content.Replace("@@autosync_repairable_faster@@", autosync_repairable_faster);
            content = content.Replace("@@gpurepair_repairable_faster@@", gpurepair_repairable_faster);

            content = content.Replace("@@autosync_time@@", autosync_time);
            content = content.Replace("@@gpurepair_time@@", gpurepair_time);
            content = content.Replace("@@autosync_median@@", autosync_median, 2);
            content = content.Replace("@@gpurepair_median@@", gpurepair_median, 2);

            content = content.Replace("@@repaired_count@@", repaired_count);
            content = content.Replace("@@unchanged_count@@", unchanged_count);
            content = content.Replace("@@repaired_unchanged@@", repaired_unchanged);

            content = content.Replace("@@autosync_both_vercalls@@", autosync_both_vercalls);
            content = content.Replace("@@gpurepair_both_vercalls@@", gpurepair_both_vercalls);
            content = content.Replace("@@autosync_repaired_vercalls@@", autosync_repaired_vercalls);
            content = content.Replace("@@gpurepair_repaired_vercalls@@", gpurepair_repaired_vercalls);
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

            IEnumerable<ToolComparisonRecord> cuda_verified =
                cuda.Where(x => x.GPUVerify_Status == "PASS");
            IEnumerable<ToolComparisonRecord> opencl_verified =
                opencl.Where(x => x.GPUVerify_Status == "PASS");

            IEnumerable<ToolComparisonRecord> cuda_verified_as_unchanged =
                cuda_verified.Where(x => x.AutoSync_Status == "UNCHANGED");
            IEnumerable<ToolComparisonRecord> cuda_verified_gr_unchanged =
                cuda_verified.Where(x => x.GPURepair_Status == "UNCHANGED");
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_unchanged =
                opencl_verified.Where(x => x.GPURepair_Status == "UNCHANGED");

            IEnumerable<ToolComparisonRecord> cuda_verified_gr_changed =
                cuda_verified.Where(x => x.GPURepair_Status == "PASS");
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_changed =
                opencl_verified.Where(x => x.GPURepair_Status == "PASS");

            IEnumerable<ToolComparisonRecord> cuda_verified_as_timeout =
                cuda_verified.Where(x => x.AutoSync_Status == "TIMEOUT");
            IEnumerable<ToolComparisonRecord> cuda_verified_gr_timeout =
                cuda_verified.Where(x => x.GPURepair_Status == "FAIL(7)");
            IEnumerable<ToolComparisonRecord> opencl_verified_gr_timeout =
                opencl_verified.Where(x => x.GPURepair_Status == "FAIL(7)");

            int cuda_verified_as_errors = cuda_verified.Count() -
                (cuda_verified_as_unchanged.Count() + cuda_verified_as_timeout.Count());
            int cuda_verified_gr_errors = cuda_verified.Count() -
                (cuda_verified_gr_unchanged.Count() + cuda_verified_gr_changed.Count() + cuda_verified_gr_timeout.Count());
            int opencl_verified_gr_errors = opencl_verified.Count() -
                (opencl_verified_gr_unchanged.Count() + opencl_verified_gr_changed.Count() + opencl_verified_gr_timeout.Count());

            IEnumerable<ToolComparisonRecord> cuda_repairable =
                cuda.Where(x => x.GPUVerify_Status == "FAIL(6)");
            IEnumerable<ToolComparisonRecord> opencl_repairable =
                opencl.Where(x => x.GPUVerify_Status == "FAIL(6)");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_repaired =
                cuda_repairable.Where(x => x.AutoSync_Status == "REPAIRED");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_repaired =
                cuda_repairable.Where(x => x.GPURepair_Status == "PASS");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_repaired =
                opencl_repairable.Where(x => x.GPURepair_Status == "PASS");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_repairerror =
                cuda_repairable.Where(x => x.AutoSync_Status == "REPAIR_ERROR");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_repairerror =
                cuda_repairable.Where(x => x.GPURepair_Status == "FAIL(202)");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_repairerror =
                opencl_repairable.Where(x => x.GPURepair_Status == "FAIL(202)");

            IEnumerable<ToolComparisonRecord> cuda_repairable_as_timeout =
                cuda_repairable.Where(x => x.AutoSync_Status == "TIMEOUT");
            IEnumerable<ToolComparisonRecord> cuda_repairable_gr_timeout =
                cuda_repairable.Where(x => x.GPURepair_Status == "FAIL(7)");
            IEnumerable<ToolComparisonRecord> opencl_repairable_gr_timeout =
                opencl_repairable.Where(x => x.GPURepair_Status == "FAIL(7)");

            int cuda_repairable_as_errors = cuda_repairable.Count() -
                (cuda_repairable_as_repaired.Count() + cuda_repairable_as_repairerror.Count() + cuda_repairable_as_timeout.Count());
            int cuda_repairable_gr_errors = cuda_repairable.Count() -
                (cuda_repairable_gr_repaired.Count() + cuda_repairable_gr_repairerror.Count() + cuda_repairable_gr_timeout.Count());
            int opencl_repairable_gr_errors = opencl_repairable.Count() -
                (opencl_repairable_gr_repaired.Count() + opencl_repairable_gr_repairerror.Count() + opencl_repairable_gr_timeout.Count());

            List<ToolComparisonRecord> cuda_repairable_gr_repaired_grid = new List<ToolComparisonRecord>();
            foreach (ToolComparisonRecord record in cuda_repairable_gr_repaired)
            {
                GPURepairRecord repair_record = FileParser.GPURepair.FirstOrDefault(x => x.Kernel == record.Kernel);
                if (repair_record != null && repair_record.SolutionGrid != 0)
                    cuda_repairable_gr_repaired_grid.Add(record);
            }

            int cuda_repairable_gr_repaired_count =
                cuda_repairable_gr_repaired.Count() - cuda_repairable_gr_repaired_grid.Count;

            IEnumerable<ToolComparisonRecord> cuda_unrepairable =
                cuda.Where(x => x.GPUVerify_Status != "PASS" && x.GPUVerify_Status != "FAIL(6)");
            IEnumerable<ToolComparisonRecord> opencl_unrepairable =
                opencl.Where(x => x.GPUVerify_Status != "PASS" && x.GPUVerify_Status != "FAIL(6)");

            IEnumerable<ToolComparisonRecord> cuda_unrepairable_as_errors =
                cuda_unrepairable.Where(x => x.AutoSync_Status == "ERROR");
            IEnumerable<ToolComparisonRecord> cuda_unrepairable_as_falsepositives =
                cuda_unrepairable.Where(x => x.AutoSync_Status == "FALSE_POSITIVE");

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "results.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@cuda@@", cuda);
            content = content.Replace("@@opencl@@", opencl);

            content = content.Replace("@@cuda_verified@@", cuda_verified);
            content = content.Replace("@@opencl_verified@@", opencl_verified);

            content = content.Replace("@@cuda_verified_as_unchanged@@", cuda_verified_as_unchanged);
            content = content.Replace("@@cuda_verified_gr_unchanged@@", cuda_verified_gr_unchanged);
            content = content.Replace("@@opencl_verified_gr_unchanged@@", opencl_verified_gr_unchanged);

            content = content.Replace("@@cuda_verified_gr_changed@@", cuda_verified_gr_changed);
            content = content.Replace("@@opencl_verified_gr_changed@@", opencl_verified_gr_changed);

            content = content.Replace("@@cuda_verified_as_timeout@@", cuda_verified_as_timeout);
            content = content.Replace("@@cuda_verified_gr_timeout@@", cuda_verified_gr_timeout);
            content = content.Replace("@@opencl_verified_gr_timeout@@", opencl_verified_gr_timeout);

            content = content.Replace("@@cuda_verified_as_errors@@", cuda_verified_as_errors);
            content = content.Replace("@@cuda_verified_gr_errors@@", cuda_verified_gr_errors);
            content = content.Replace("@@opencl_verified_gr_errors@@", opencl_verified_gr_errors);

            content = content.Replace("@@cuda_repairable@@", cuda_repairable);
            content = content.Replace("@@opencl_repairable@@", opencl_repairable);

            content = content.Replace("@@cuda_repairable_as_repaired@@", cuda_repairable_as_repaired);
            content = content.Replace("@@cuda_repairable_gr_repaired@@", cuda_repairable_gr_repaired_count);
            content = content.Replace("@@opencl_repairable_gr_repaired@@", opencl_repairable_gr_repaired);

            content = content.Replace("@@cuda_repairable_gr_repaired_grid@@", cuda_repairable_gr_repaired_grid);

            content = content.Replace("@@cuda_repairable_as_repairerror@@", cuda_repairable_as_repairerror);
            content = content.Replace("@@cuda_repairable_gr_repairerror@@", cuda_repairable_gr_repairerror);
            content = content.Replace("@@opencl_repairable_gr_repairerror@@", opencl_repairable_gr_repairerror);

            content = content.Replace("@@cuda_repairable_as_timeout@@", cuda_repairable_as_timeout);
            content = content.Replace("@@cuda_repairable_gr_timeout@@", cuda_repairable_gr_timeout);
            content = content.Replace("@@opencl_repairable_gr_timeout@@", opencl_repairable_gr_timeout);

            content = content.Replace("@@cuda_repairable_as_errors@@", cuda_repairable_as_errors);
            content = content.Replace("@@cuda_repairable_gr_errors@@", cuda_repairable_gr_errors);
            content = content.Replace("@@opencl_repairable_gr_errors@@", opencl_repairable_gr_errors);

            content = content.Replace("@@cuda_unrepairable@@", cuda_unrepairable);
            content = content.Replace("@@opencl_unrepairable@@", opencl_unrepairable);

            content = content.Replace("@@cuda_unrepairable_as_errors@@", cuda_unrepairable_as_errors);
            content = content.Replace("@@cuda_unrepairable_as_falsepositives@@", cuda_unrepairable_as_falsepositives);
            File.WriteAllText(file, content);
        }

        private static void PrepareToolTimeGraphs(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_repaired.dat";

            IEnumerable<string> data = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.GPUVerify_Status == "FAIL(6)")
                .Where(x => x.AutoSync_Time > 0 && x.GPURepair_Time > 0)
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GPURepair_Time, 2).ToString(),
                    Math.Round(x.AutoSync_Time, 2).ToString(),
                    "a"
                }));

            string content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_scatter.dat";

            data = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.GPUVerify_Status == "FAIL(6)" || x.GPUVerify_Status == "PASS")
                .Where(x => x.AutoSync_Time > 0 && x.GPURepair_Time > 0)
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GPURepair_Time, 2).ToString(),
                    Math.Round(x.AutoSync_Time, 2).ToString(),
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
                .Where(x => x.AutoSync_Status == "REPAIRED" && x.GPURepair_Status == "PASS")
                .Where(x => x.AutoSync_VerCount > 0 && x.GPURepair_VerCount > 0);

            IEnumerable<ToolComparisonRecord> unchanged = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.AutoSync_Status == "UNCHANGED" && x.GPURepair_Status == "UNCHANGED")
                .Where(x => x.AutoSync_VerCount > 0 && x.GPURepair_VerCount > 0);

            List<string> data = new List<string>();
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                {
                    // calls more than 6 are considered as 6
                    int count = repaired.Union(unchanged).Count(x =>
                        (Convert.ToInt32(Math.Round(x.GPURepair_VerCount > 6 ? 6 : x.GPURepair_VerCount)) == (i + 1)) &&
                        (Convert.ToInt32(Math.Round(x.AutoSync_VerCount > 6 ? 6 : x.AutoSync_VerCount)) == (j + 1))
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
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                {
                    // calls more than 6 are considered as 6
                    int count = repaired.Count(x =>
                        (Convert.ToInt32(Math.Round(x.GPURepair_VerCount > 6 ? 6 : x.GPURepair_VerCount)) == (i + 1)) &&
                        (Convert.ToInt32(Math.Round(x.AutoSync_VerCount > 6 ? 6 : x.AutoSync_VerCount)) == (j + 1))
                    );

                    if (count != 0)
                        data.Add(string.Join("\t", new string[] { (i + 1).ToString(), (j + 1).ToString(),
                            (count > 5 ? 5 + (count / 10) : count).ToString(), count.ToString() }));
                }

            content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);
        }

        private static void PrepareSourceInformation(string directory)
        {
            int kernel_count = FileParser.GPURepair.Count();
            double average_lines = FileParser.GPURepair.Select(x => x.Lines).Average();
            double median_lines = Median(FileParser.GPURepair.Select(x => x.Lines));
            int above_hundred = FileParser.GPURepair.Count(x => x.Lines >= 100);
            int above_fifty = FileParser.GPURepair.Count(x => x.Lines >= 50);
            int max_lines = Convert.ToInt32(FileParser.GPURepair.Max(x => x.Lines));

            IEnumerable<GPURepairRecord> instrumentation = FileParser.GPURepair.Where(x => x.Instrumentation > 0);
            int instrumentation_count = instrumentation.Count();

            double average_commands = instrumentation.Select(x => x.Commands).Average();
            double median_commands = Median(instrumentation.Select(x => x.Commands));
            int max_commands = Convert.ToInt32(instrumentation.Max(x => x.Commands));

            double below_three = (instrumentation.Count(x => x.Barriers <= 3) * 100.0) / instrumentation_count;
            int above_hundred_barriers = instrumentation.Count(x => x.Barriers >= 100);
            double time_second = (instrumentation.Count(x => x.Instrumentation <= 1) * 100.0) / instrumentation_count;

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "source_information.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", kernel_count);
            content = content.Replace("@@average_lines@@", average_lines, 2);
            content = content.Replace("@@median_lines@@", median_lines, 2);
            content = content.Replace("@@above_hundred@@", above_hundred);
            content = content.Replace("@@above_fifty@@", above_fifty);
            content = content.Replace("@@max_lines@@", max_lines);

            content = content.Replace("@@average_commands@@", average_commands);
            content = content.Replace("@@median_commands@@", median_commands);
            content = content.Replace("@@max_commands@@", max_commands);

            content = content.Replace("@@instrumentation_count@@", instrumentation_count);
            content = content.Replace("@@below_three@@", below_three, 2);
            content = content.Replace("@@above_hundred_barriers@@", above_hundred_barriers);
            content = content.Replace("@@time_second@@", time_second, 2);
            File.WriteAllText(file, content);

            PrepareSourceGraph(directory);
        }

        private static void PrepareSourceGraph(string directory)
        {
            int lines_10 = FileParser.GPURepair.Count(x => x.Lines <= 10);
            int lines_25 = FileParser.GPURepair.Count(x => x.Lines > 10 && x.Lines <= 25);
            int lines_50 = FileParser.GPURepair.Count(x => x.Lines > 25 && x.Lines <= 50);
            int lines_75 = FileParser.GPURepair.Count(x => x.Lines > 50 && x.Lines <= 75);
            int lines_100 = FileParser.GPURepair.Count(x => x.Lines > 75 && x.Lines <= 100);
            int lines_inf = FileParser.GPURepair.Count(x => x.Lines > 100);

            int commands_10 = FileParser.GPURepair.Count(x => x.Commands <= 10);
            int commands_25 = FileParser.GPURepair.Count(x => x.Commands > 10 && x.Commands <= 25);
            int commands_50 = FileParser.GPURepair.Count(x => x.Commands > 25 && x.Commands <= 50);
            int commands_75 = FileParser.GPURepair.Count(x => x.Commands > 50 && x.Commands <= 75);
            int commands_100 = FileParser.GPURepair.Count(x => x.Commands > 75 && x.Commands <= 100);
            int commands_inf = FileParser.GPURepair.Count(x => x.Commands > 100);

            IEnumerable<GPURepairRecord> instrumentation = FileParser.GPURepair.Where(x => x.Instrumentation > 0);

            int barriers_0 = instrumentation.Count(x => Math.Round(x.Barriers) == 0);
            int barriers_1 = instrumentation.Count(x => Math.Round(x.Barriers) == 1);
            int barriers_2 = instrumentation.Count(x => Math.Round(x.Barriers) == 2);
            int barriers_3 = instrumentation.Count(x => Math.Round(x.Barriers) == 3);
            int barriers_5 = instrumentation.Count(x => x.Barriers >= 4 && x.Barriers <= 5);
            int barriers_10 = instrumentation.Count(x => x.Barriers > 5 && x.Barriers <= 10);
            int barriers_20 = instrumentation.Count(x => x.Barriers > 10 && x.Barriers <= 20);
            int barriers_100 = instrumentation.Count(x => x.Barriers > 20 && x.Barriers <= 100);
            int barriers_inf = instrumentation.Count(x => x.Barriers > 100);

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "lines_commands.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@lines_10@@", lines_10);
            content = content.Replace("@@lines_25@@", lines_25);
            content = content.Replace("@@lines_50@@", lines_50);
            content = content.Replace("@@lines_75@@", lines_75);
            content = content.Replace("@@lines_100@@", lines_100);
            content = content.Replace("@@lines_inf@@", lines_inf);

            content = content.Replace("@@commands_10@@", commands_10);
            content = content.Replace("@@commands_25@@", commands_25);
            content = content.Replace("@@commands_50@@", commands_50);
            content = content.Replace("@@commands_75@@", commands_75);
            content = content.Replace("@@commands_100@@", commands_100);
            content = content.Replace("@@commands_inf@@", commands_inf);
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "instrumentation.tex";

            content = File.ReadAllText(file);
            content = content.Replace("@@barriers_0@@", barriers_0);
            content = content.Replace("@@barriers_1@@", barriers_1);
            content = content.Replace("@@barriers_2@@", barriers_2);
            content = content.Replace("@@barriers_3@@", barriers_3);
            content = content.Replace("@@barriers_5@@", barriers_5);
            content = content.Replace("@@barriers_10@@", barriers_10);
            content = content.Replace("@@barriers_20@@", barriers_20);
            content = content.Replace("@@barriers_100@@", barriers_100);
            content = content.Replace("@@barriers_inf@@", barriers_inf);
            File.WriteAllText(file, content);
        }

        private static void PrepareSolverComparison(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "solver_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", DataAnalyzer.SolverComparisonRecords);
            File.WriteAllText(file, content);

            PrepareSolverGraphs(directory);
        }

        private static void PrepareSolverGraphs(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_mhs_maxsat.dat";

            string content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n",
                DataAnalyzer.SolverComparisonRecords.Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.mhs_Time, 2).ToString(),
                    Math.Round(x.MaxSAT_Time, 2).ToString(),
                    "a"
                }))));

            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_mhs_sat.dat";

            content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n",
                DataAnalyzer.SolverComparisonRecords.Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.mhs_Time, 2).ToString(),
                    Math.Round(x.SAT_Time, 2).ToString(),
                    "a"
                }))));

            File.WriteAllText(file, content);
        }

        private static void PrepareConfigurationComparison(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "configuration_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", DataAnalyzer.ConfigurationComparisonRecords);
            File.WriteAllText(file, content);

            PrepareConfigurationTables(directory);
            PrepareConfigurationAutoSyncTables(directory);
        }

        private static void PrepareConfigurationTables(string directory)
        {
            int default_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.GPURepair_Status == "PASS");
            int default_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.GPURepair_Status == "PASS");
            int default_total = default_repaired + default_optimized;
            double default_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.GPURepair_Time);
            double default_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.GPURepair_Time));
            int default_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.GPURepair_SolverCount)));
            int default_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.GPURepair_VerCount)));

            int maxsat_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.MaxSAT_Status == "PASS");
            int maxsat_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.MaxSAT_Status == "PASS");
            int maxsat_total = maxsat_repaired + maxsat_optimized;
            double maxsat_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.MaxSAT_Time);
            double maxsat_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.MaxSAT_Time));
            int maxsat_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.MaxSAT_SolverCount)));
            int maxsat_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.MaxSAT_VerCount)));

            int sat_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.SAT_Status == "PASS");
            int sat_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.SAT_Status == "PASS");
            int sat_total = sat_repaired + sat_optimized;
            double sat_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.SAT_Time);
            double sat_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.SAT_Time));
            int sat_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.SAT_SolverCount)));
            int sat_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.SAT_VerCount)));

            int dg_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.Grid_Status == "PASS");
            int dg_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.Grid_Status == "PASS");
            int dg_total = dg_repaired + dg_optimized;
            double dg_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_Time);
            double dg_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.Grid_Time));
            int dg_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_SolverCount)));
            int dg_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_VerCount)));

            int di_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.Inspection_Status == "PASS");
            int di_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.Inspection_Status == "PASS");
            int di_total = di_repaired + di_optimized;
            double di_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Inspection_Time);
            double di_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.Inspection_Time));
            int di_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Inspection_SolverCount)));
            int di_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Inspection_VerCount)));

            int dg_di_repaired =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "FAIL(6)" && x.Grid_Inspection_Status == "PASS");
            int dg_di_optimized =
                DataAnalyzer.ConfigurationComparisonRecords.Count(x => x.GPUVerify_Status == "PASS" && x.Grid_Inspection_Status == "PASS");
            int dg_di_total = dg_di_repaired + dg_di_optimized;
            double dg_di_time = DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_Inspection_Time);
            double dg_di_median = Median(DataAnalyzer.ConfigurationComparisonRecords.Select(x => x.Grid_Inspection_Time));
            int dg_di_solver = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_Inspection_SolverCount)));
            int dg_di_verifier = Convert.ToInt32(Math.Round(DataAnalyzer.ConfigurationComparisonRecords.Sum(x => x.Grid_Inspection_VerCount)));

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "configuration_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@default_repaired@@", default_repaired);
            content = content.Replace("@@default_optimized@@", default_optimized);
            content = content.Replace("@@default_total@@", default_total);
            content = content.Replace("@@default_time@@", default_time);
            content = content.Replace("@@default_median@@", default_median, 2);
            content = content.Replace("@@default_solver@@", default_solver);
            content = content.Replace("@@default_verifier@@", default_verifier);

            content = content.Replace("@@maxsat_repaired@@", maxsat_repaired);
            content = content.Replace("@@maxsat_optimized@@", maxsat_optimized);
            content = content.Replace("@@maxsat_total@@", maxsat_total);
            content = content.Replace("@@maxsat_time@@", maxsat_time);
            content = content.Replace("@@maxsat_median@@", maxsat_median, 2);
            content = content.Replace("@@maxsat_solver@@", maxsat_solver);
            content = content.Replace("@@maxsat_verifier@@", maxsat_verifier);

            content = content.Replace("@@sat_repaired@@", sat_repaired);
            content = content.Replace("@@sat_optimized@@", sat_optimized);
            content = content.Replace("@@sat_total@@", sat_total);
            content = content.Replace("@@sat_time@@", sat_time);
            content = content.Replace("@@sat_median@@", sat_median, 2);
            content = content.Replace("@@sat_solver@@", sat_solver);
            content = content.Replace("@@sat_verifier@@", sat_verifier);

            content = content.Replace("@@dg_repaired@@", dg_repaired);
            content = content.Replace("@@dg_optimized@@", dg_optimized);
            content = content.Replace("@@dg_total@@", dg_total);
            content = content.Replace("@@dg_time@@", dg_time);
            content = content.Replace("@@dg_median@@", dg_median, 2);
            content = content.Replace("@@dg_solver@@", dg_solver);
            content = content.Replace("@@dg_verifier@@", dg_verifier);

            content = content.Replace("@@di_repaired@@", di_repaired);
            content = content.Replace("@@di_optimized@@", di_optimized);
            content = content.Replace("@@di_total@@", di_total);
            content = content.Replace("@@di_time@@", di_time);
            content = content.Replace("@@di_median@@", di_median, 2);
            content = content.Replace("@@di_solver@@", di_solver);
            content = content.Replace("@@di_verifier@@", di_verifier);

            content = content.Replace("@@dg_di_repaired@@", dg_di_repaired);
            content = content.Replace("@@dg_di_optimized@@", dg_di_optimized);
            content = content.Replace("@@dg_di_total@@", dg_di_total);
            content = content.Replace("@@dg_di_time@@", dg_di_time);
            content = content.Replace("@@dg_di_median@@", dg_di_median, 2);
            content = content.Replace("@@dg_di_solver@@", dg_di_solver);
            content = content.Replace("@@dg_di_verifier@@", dg_di_verifier);
            File.WriteAllText(file, content);
        }

        private static void PrepareConfigurationAutoSyncTables(string directory)
        {
            IEnumerable<ConfigurationComparisonRecord> cuda_valid =
                DataAnalyzer.ConfigurationComparisonRecords.Where(x => !x.Kernel.Contains("OpenCL"))
                .Where(x => x.GPUVerify_Status == "PASS" || x.GPUVerify_Status == "FAIL(6)");
            IEnumerable<ConfigurationComparisonRecord> cuda_valid_same =
                cuda_valid.Where(x => x.GPURepair_Status == x.MaxSAT_Status &&
                    x.GPURepair_Status == x.SAT_Status &&
                    x.GPURepair_Status == x.Grid_Status &&
                    x.GPURepair_Status == x.Inspection_Status &&
                    x.GPURepair_Status == x.Grid_Inspection_Status);

            IEnumerable<ConfigurationComparisonRecord> cuda_repaired =
                cuda_valid_same.Where(x => x.GPUVerify_Status == "FAIL(6)" && x.AutoSync_Status == "REPAIRED" && x.GPURepair_Status == "PASS");
            IEnumerable<ConfigurationComparisonRecord> cuda_unchanged =
                cuda_valid_same.Where(x => x.GPUVerify_Status == "PASS" && x.AutoSync_Status == "UNCHANGED" && x.GPURepair_Status == "UNCHANGED");
            IEnumerable<ConfigurationComparisonRecord> cuda_repaired_unchanged = cuda_repaired.Union(cuda_unchanged);

            double as_time = cuda_valid.Sum(x => x.AutoSync_Time);
            double as_median = Median(cuda_valid.Select(x => x.AutoSync_Time));
            double as_ru_time = cuda_repaired_unchanged.Sum(x => x.AutoSync_Time);
            double as_ru_median = Median(cuda_repaired_unchanged.Select(x => x.AutoSync_Time));
            int as_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.AutoSync_VerCount)));

            double default_time = cuda_valid.Sum(x => x.GPURepair_Time);
            double default_median = Median(cuda_valid.Select(x => x.GPURepair_Time));
            double default_ru_time = cuda_repaired_unchanged.Sum(x => x.GPURepair_Time);
            double default_ru_median = Median(cuda_repaired_unchanged.Select(x => x.GPURepair_Time));
            int default_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.GPURepair_VerCount)));

            double maxsat_time = cuda_valid.Sum(x => x.MaxSAT_Time);
            double maxsat_median = Median(cuda_valid.Select(x => x.MaxSAT_Time));
            double maxsat_ru_time = cuda_repaired_unchanged.Sum(x => x.MaxSAT_Time);
            double maxsat_ru_median = Median(cuda_repaired_unchanged.Select(x => x.MaxSAT_Time));
            int maxsat_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.MaxSAT_VerCount)));

            double sat_time = cuda_valid.Sum(x => x.SAT_Time);
            double sat_median = Median(cuda_valid.Select(x => x.SAT_Time));
            double sat_ru_time = cuda_repaired_unchanged.Sum(x => x.SAT_Time);
            double sat_ru_median = Median(cuda_repaired_unchanged.Select(x => x.SAT_Time));
            int sat_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.SAT_VerCount)));

            double grid_time = cuda_valid.Sum(x => x.Grid_Time);
            double grid_median = Median(cuda_valid.Select(x => x.Grid_Time));
            double grid_ru_time = cuda_repaired_unchanged.Sum(x => x.Grid_Time);
            double grid_ru_median = Median(cuda_repaired_unchanged.Select(x => x.Grid_Time));
            int grid_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.Grid_VerCount)));

            double inspection_time = cuda_valid.Sum(x => x.Inspection_Time);
            double inspection_median = Median(cuda_valid.Select(x => x.Inspection_Time));
            double inspection_ru_time = cuda_repaired_unchanged.Sum(x => x.Inspection_Time);
            double inspection_ru_median = Median(cuda_repaired_unchanged.Select(x => x.Inspection_Time));
            int inspection_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.Inspection_VerCount)));

            double grid_inspection_time = cuda_valid.Sum(x => x.Grid_Inspection_Time);
            double grid_inspection_median = Median(cuda_valid.Select(x => x.Grid_Inspection_Time));
            double grid_inspection_ru_time = cuda_repaired_unchanged.Sum(x => x.Grid_Inspection_Time);
            double grid_inspection_ru_median = Median(cuda_repaired_unchanged.Select(x => x.Grid_Inspection_Time));
            int grid_inspection_ru_verifier = Convert.ToInt32(Math.Round(cuda_repaired_unchanged.Sum(x => x.Grid_Inspection_VerCount)));

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + "configuration_comparison_autosync.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", cuda_valid);
            content = content.Replace("@@repaired@@", cuda_repaired);
            content = content.Replace("@@unchanged@@", cuda_unchanged);
            content = content.Replace("@@total@@", cuda_repaired.Count() + cuda_unchanged.Count());

            content = content.Replace("@@as_time@@", as_time);
            content = content.Replace("@@as_median@@", as_median, 2);
            content = content.Replace("@@as_ru_time@@", as_ru_time);
            content = content.Replace("@@as_ru_median@@", as_ru_median, 2);
            content = content.Replace("@@as_ru_verifier@@", as_ru_verifier);

            content = content.Replace("@@default_time@@", default_time);
            content = content.Replace("@@default_median@@", default_median, 2);
            content = content.Replace("@@default_ru_time@@", default_ru_time);
            content = content.Replace("@@default_ru_median@@", default_ru_median, 2);
            content = content.Replace("@@default_ru_verifier@@", default_ru_verifier);

            content = content.Replace("@@maxsat_time@@", maxsat_time);
            content = content.Replace("@@maxsat_median@@", maxsat_median, 2);
            content = content.Replace("@@maxsat_ru_time@@", maxsat_ru_time);
            content = content.Replace("@@maxsat_ru_median@@", maxsat_ru_median, 2);
            content = content.Replace("@@maxsat_ru_verifier@@", maxsat_ru_verifier);

            content = content.Replace("@@sat_time@@", sat_time);
            content = content.Replace("@@sat_median@@", sat_median, 2);
            content = content.Replace("@@sat_ru_time@@", sat_ru_time);
            content = content.Replace("@@sat_ru_median@@", sat_ru_median, 2);
            content = content.Replace("@@sat_ru_verifier@@", sat_ru_verifier);

            content = content.Replace("@@grid_time@@", grid_time);
            content = content.Replace("@@grid_median@@", grid_median, 2);
            content = content.Replace("@@grid_ru_time@@", grid_ru_time);
            content = content.Replace("@@grid_ru_median@@", grid_ru_median, 2);
            content = content.Replace("@@grid_ru_verifier@@", grid_ru_verifier);

            content = content.Replace("@@inspection_time@@", inspection_time);
            content = content.Replace("@@inspection_median@@", inspection_median, 2);
            content = content.Replace("@@inspection_ru_time@@", inspection_ru_time);
            content = content.Replace("@@inspection_ru_median@@", inspection_ru_median, 2);
            content = content.Replace("@@inspection_ru_verifier@@", inspection_ru_verifier);

            content = content.Replace("@@grid_inspection_time@@", grid_inspection_time);
            content = content.Replace("@@grid_inspection_median@@", grid_inspection_median, 2);
            content = content.Replace("@@grid_inspection_ru_time@@", grid_inspection_ru_time);
            content = content.Replace("@@grid_inspection_ru_median@@", grid_inspection_ru_median, 2);
            content = content.Replace("@@grid_inspection_ru_verifier@@", grid_inspection_ru_verifier);
            File.WriteAllText(file, content);
        }

        private static double Median(IEnumerable<double> values)
        {
            int count = values.Count();
            int index = count / 2;

            IEnumerable<double> sorted = values.OrderBy(n => n);
            double median = count % 2 != 0 ? sorted.ElementAt(index) :
                (sorted.ElementAt(index) + sorted.ElementAt(index - 1)) / 2;

            return median;
        }
    }
}
