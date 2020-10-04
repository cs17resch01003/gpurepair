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

            double autosync_time = valid.Sum(x => x.AS_Time);
            double gpurepair_time = valid.Sum(x => x.GR_Time);
            double autosync_median = Median(valid.Select(x => x.AS_Time));
            double gpurepair_median = Median(valid.Select(x => x.GR_Time));

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

            content = content.Replace("@@autosync_both_vercalls@@", Convert.ToInt32(Math.Round(autosync_both_vercalls)).ToString());
            content = content.Replace("@@gpurepair_both_vercalls@@", Convert.ToInt32(Math.Round(gpurepair_both_vercalls)).ToString());
            content = content.Replace("@@autosync_repaired_vercalls@@", Convert.ToInt32(Math.Round(autosync_repaired_vercalls)).ToString());
            content = content.Replace("@@gpurepair_repaired_vercalls@@", Convert.ToInt32(Math.Round(gpurepair_repaired_vercalls)).ToString());
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
            content = content.Replace("@@cuda@@", cuda.Count().ToString());
            content = content.Replace("@@opencl@@", opencl.Count().ToString());

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

            content = content.Replace("@@cuda_repairable_as_repairerror@@", cuda_repairable_as_repairerror.Count().ToString());
            content = content.Replace("@@cuda_repairable_gr_repairerror@@", cuda_repairable_gr_repairerror.Count().ToString());
            content = content.Replace("@@opencl_repairable_gr_repairerror@@", opencl_repairable_gr_repairerror.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_timeout@@", cuda_repairable_as_timeout.Count().ToString());
            content = content.Replace("@@cuda_repairable_gr_timeout@@", cuda_repairable_gr_timeout.Count().ToString());
            content = content.Replace("@@opencl_repairable_gr_timeout@@", opencl_repairable_gr_timeout.Count().ToString());

            content = content.Replace("@@cuda_repairable_as_errors@@", cuda_repairable_as_errors.ToString());
            content = content.Replace("@@cuda_repairable_gr_errors@@", cuda_repairable_gr_errors.ToString());
            content = content.Replace("@@opencl_repairable_gr_errors@@", opencl_repairable_gr_errors.ToString());

            content = content.Replace("@@cuda_unrepairable@@", cuda_unrepairable.Count().ToString());
            content = content.Replace("@@opencl_unrepairable@@", opencl_unrepairable.Count().ToString());

            content = content.Replace("@@cuda_unrepairable_as_errors@@", cuda_unrepairable_as_errors.Count().ToString());
            content = content.Replace("@@cuda_unrepairable_as_falsepositives@@", cuda_unrepairable_as_falsepositives.Count().ToString());
            File.WriteAllText(file, content);
        }

        private static void PrepareToolTimeGraphs(string directory)
        {
            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_repaired.dat";

            IEnumerable<string> data = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.GV_Status == "FAIL(6)")
                .Where(x => x.AS_Time > 0 && x.GR_Time > 0)
                .Select(x => string.Join("\t", new string[]
                {
                    Math.Round(x.GR_Time, 2).ToString(),
                    Math.Round(x.AS_Time, 2).ToString(),
                    "a"
                }));

            string content = File.ReadAllText(file);
            content = content.Replace("@@data@@", string.Join("\r\n", data));
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "data" + Path.DirectorySeparatorChar + "time_scatter.dat";

            data = DataAnalyzer.ToolComparisonRecords
                .Where(x => x.GV_Status == "FAIL(6)" || x.GV_Status == "PASS")
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
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                {
                    // calls more than 6 are considered as 6
                    int count = repaired.Union(unchanged).Count(x =>
                        (Convert.ToInt32(Math.Round(x.GR_VerCount > 6 ? 6 : x.GR_VerCount)) == (i + 1)) &&
                        (Convert.ToInt32(Math.Round(x.AS_VerCount > 6 ? 6 : x.AS_VerCount)) == (j + 1))
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
                        (Convert.ToInt32(Math.Round(x.GR_VerCount > 6 ? 6 : x.GR_VerCount)) == (i + 1)) &&
                        (Convert.ToInt32(Math.Round(x.AS_VerCount > 6 ? 6 : x.AS_VerCount)) == (j + 1))
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
            content = content.Replace("@@kernel_count@@", kernel_count.ToString());
            content = content.Replace("@@average_lines@@", Math.Round(average_lines, 2).ToString());
            content = content.Replace("@@median_lines@@", Math.Round(median_lines, 2).ToString());
            content = content.Replace("@@above_hundred@@", above_hundred.ToString());
            content = content.Replace("@@above_fifty@@", above_fifty.ToString());
            content = content.Replace("@@max_lines@@", max_lines.ToString());

            content = content.Replace("@@average_commands@@", Math.Round(average_commands, 2).ToString());
            content = content.Replace("@@median_commands@@", Math.Round(median_commands, 2).ToString());
            content = content.Replace("@@max_commands@@", max_commands.ToString());

            content = content.Replace("@@instrumentation_count@@", instrumentation_count.ToString());
            content = content.Replace("@@below_three@@", Math.Round(below_three, 2).ToString());
            content = content.Replace("@@above_hundred_barriers@@", above_hundred_barriers.ToString());
            content = content.Replace("@@time_second@@", Math.Round(time_second, 2).ToString());
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
            content = content.Replace("@@lines_10@@", lines_10.ToString());
            content = content.Replace("@@lines_25@@", lines_25.ToString());
            content = content.Replace("@@lines_50@@", lines_50.ToString());
            content = content.Replace("@@lines_75@@", lines_75.ToString());
            content = content.Replace("@@lines_100@@", lines_100.ToString());
            content = content.Replace("@@lines_inf@@", lines_inf.ToString());

            content = content.Replace("@@commands_10@@", commands_10.ToString());
            content = content.Replace("@@commands_25@@", commands_25.ToString());
            content = content.Replace("@@commands_50@@", commands_50.ToString());
            content = content.Replace("@@commands_75@@", commands_75.ToString());
            content = content.Replace("@@commands_100@@", commands_100.ToString());
            content = content.Replace("@@commands_inf@@", commands_inf.ToString());
            File.WriteAllText(file, content);

            file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "figures" + Path.DirectorySeparatorChar + "instrumentation.tex";

            content = File.ReadAllText(file);
            content = content.Replace("@@barriers_0@@", barriers_0.ToString());
            content = content.Replace("@@barriers_1@@", barriers_1.ToString());
            content = content.Replace("@@barriers_2@@", barriers_2.ToString());
            content = content.Replace("@@barriers_3@@", barriers_3.ToString());
            content = content.Replace("@@barriers_5@@", barriers_5.ToString());
            content = content.Replace("@@barriers_10@@", barriers_10.ToString());
            content = content.Replace("@@barriers_20@@", barriers_20.ToString());
            content = content.Replace("@@barriers_100@@", barriers_100.ToString());
            content = content.Replace("@@barriers_inf@@", barriers_inf.ToString());
            File.WriteAllText(file, content);
        }

        private static void PrepareSolverComparison(string directory)
        {
            int kernel_count = DataAnalyzer.SolverComparisonRecords.Count();

            IEnumerable<SolverComparisonRecord> repaired = DataAnalyzer.SolverComparisonRecords
                .Where(x => x.GV_Status == "FAIL(6)" && x.mhs_Status == "PASS" && x.MaxSAT_Status == "PASS" && x.SAT_Status == "PASS");
            IEnumerable<SolverComparisonRecord> repaired_cuda = repaired.Where(x => !x.Kernel.Contains("OpenCL"));
            IEnumerable<SolverComparisonRecord> repaired_opencl = repaired.Where(x => x.Kernel.Contains("OpenCL"));

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "solver_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", kernel_count.ToString());
            content = content.Replace("@@repaired@@", repaired.Count().ToString());
            content = content.Replace("@@repaired_cuda@@", repaired_cuda.Count().ToString());
            content = content.Replace("@@repaired_opencl@@", repaired_opencl.Count().ToString());
            File.WriteAllText(file, content);

            PrepareSolverTables(directory);
            PrepareSolverGraphs(directory);
        }

        private static void PrepareSolverTables(string directory)
        {
            IEnumerable<SolverComparisonRecord> repaired = DataAnalyzer.SolverComparisonRecords
                .Where(x => x.GV_Status == "FAIL(6)" && x.mhs_Status == "PASS" && x.MaxSAT_Status == "PASS" && x.SAT_Status == "PASS");

            PrepareSolverTables(DataAnalyzer.SolverComparisonRecords, directory, "solver_comparison.tex");
            PrepareSolverTables(repaired, directory, "solver_comparison_repaired.tex");
        }

        private static void PrepareSolverTables(IEnumerable<SolverComparisonRecord> records, string directory, string filename)
        {
            double mhs_total = records.Sum(x => x.mhs_Time);
            double mhs_median = Median(records.Select(x => x.mhs_Time));
            int mhs_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.mhs_SolverCount)));
            int mhs_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.mhs_VerCount)));

            double MaxSAT_total = records.Sum(x => x.MaxSAT_Time);
            double MaxSAT_median = Median(records.Select(x => x.MaxSAT_Time));
            int MaxSAT_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.MaxSAT_SolverCount)));
            int MaxSAT_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.MaxSAT_VerCount)));

            double SAT_total = records.Sum(x => x.SAT_Time);
            double SAT_median = Median(records.Select(x => x.SAT_Time));
            int SAT_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.SAT_SolverCount)));
            int SAT_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.SAT_VerCount)));

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + filename;

            string content = File.ReadAllText(file);
            content = content.Replace("@@mhs_total@@", Math.Round(mhs_total, 2).ToString());
            content = content.Replace("@@mhs_median@@", Math.Round(mhs_median, 2).ToString());
            content = content.Replace("@@mhs_solver@@", mhs_solver.ToString());
            content = content.Replace("@@mhs_verifier@@", mhs_verifier.ToString());

            content = content.Replace("@@MaxSAT_total@@", Math.Round(MaxSAT_total, 2).ToString());
            content = content.Replace("@@MaxSAT_median@@", Math.Round(MaxSAT_median, 2).ToString());
            content = content.Replace("@@MaxSAT_solver@@", MaxSAT_solver.ToString());
            content = content.Replace("@@MaxSAT_verifier@@", MaxSAT_verifier.ToString());

            content = content.Replace("@@SAT_total@@", Math.Round(SAT_total, 2).ToString());
            content = content.Replace("@@SAT_median@@", Math.Round(SAT_median, 2).ToString());
            content = content.Replace("@@SAT_solver@@", SAT_solver.ToString());
            content = content.Replace("@@SAT_verifier@@", SAT_verifier.ToString());
            File.WriteAllText(file, content);
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
            int kernel_count = DataAnalyzer.ConfigurationComparisonRecords.Count();
            int same_count = DataAnalyzer.ConfigurationComparisonSameRecords.Count();

            IEnumerable<ConfigurationComparisonRecord> same_cuda =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x => !x.Kernel.Contains("OpenCL"));
            IEnumerable<ConfigurationComparisonRecord> same_opencl =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x => x.Kernel.Contains("OpenCL"));
            IEnumerable<ConfigurationComparisonRecord> autosync_same =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x =>
                {
                    bool unchanged = x.AutoSync_Status == "UNCHANGED" && (x.Default_Status == "PASS" && x.Default_Changes == 0);
                    bool repaired = x.GV_Status == "FAIL(6)" && x.AutoSync_Status == "REPAIRED" && x.Default_Status == "PASS";

                    return unchanged || repaired;
                });
            IEnumerable<ConfigurationComparisonRecord> autosync_repaired_same =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x =>
                x.GV_Status == "FAIL(6)" && x.AutoSync_Status == "REPAIRED" && x.Default_Status == "PASS");

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "experiments" + Path.DirectorySeparatorChar + "configuration_comparison.tex";

            string content = File.ReadAllText(file);
            content = content.Replace("@@kernel_count@@", kernel_count.ToString());
            content = content.Replace("@@same@@", same_count.ToString());
            content = content.Replace("@@same_cuda@@", same_cuda.Count().ToString());
            content = content.Replace("@@same_opencl@@", same_opencl.Count().ToString());
            content = content.Replace("@@autosync_same@@", autosync_same.Count().ToString());
            content = content.Replace("@@autosync_repaired_same@@", autosync_repaired_same.Count().ToString());
            File.WriteAllText(file, content);

            PrepareConfigurationTables(directory);
        }

        private static void PrepareConfigurationTables(string directory)
        {
            IEnumerable<ConfigurationComparisonRecord> autosync_same =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x =>
                {
                    bool unchanged = x.AutoSync_Status == "UNCHANGED" && (x.Default_Status == "PASS" && x.Default_Changes == 0);
                    bool repaired = x.GV_Status == "FAIL(6)" && x.AutoSync_Status == "REPAIRED" && x.Default_Status == "PASS";

                    return unchanged || repaired;
                });
            IEnumerable<ConfigurationComparisonRecord> autosync_repaired_same =
                DataAnalyzer.ConfigurationComparisonSameRecords.Where(x =>
                x.GV_Status == "FAIL(6)" && x.AutoSync_Status == "REPAIRED" && x.Default_Status == "PASS");

            PrepareConfigurationTables(DataAnalyzer.ConfigurationComparisonRecords, directory, "configuration_comparison.tex");
            PrepareConfigurationTables(DataAnalyzer.ConfigurationComparisonSameRecords, directory, "configuration_comparison_same.tex");
            PrepareConfigurationTables(autosync_same, directory, "configuration_comparison_autosync.tex");
            PrepareConfigurationTables(autosync_repaired_same, directory, "configuration_comparison_autosync_repaired.tex");
        }

        private static void PrepareConfigurationTables(IEnumerable<ConfigurationComparisonRecord> records, string directory, string filename)
        {
            double autosync_time = records.Sum(x => x.AutoSync_Time);
            double autosync_median = Median(records.Select(x => x.AutoSync_Time));
            int autosync_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.AutoSync_VerCount)));

            double default_time = records.Sum(x => x.Default_Time);
            double default_median = Median(records.Select(x => x.Default_Time));
            int default_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.Default_SolverCount)));
            int default_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.Default_VerCount)));

            double dg_time = records.Sum(x => x.DG_Time);
            double dg_median = Median(records.Select(x => x.DG_Time));
            int dg_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.DG_SolverCount)));
            int dg_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.DG_VerCount)));

            double di_time = records.Sum(x => x.DI_Time);
            double di_median = Median(records.Select(x => x.DI_Time));
            int di_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.DI_SolverCount)));
            int di_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.DI_VerCount)));

            double dg_di_time = records.Sum(x => x.DG_DI_Time);
            double dg_di_median = Median(records.Select(x => x.DG_DI_Time));
            int dg_di_solver = Convert.ToInt32(Math.Round(records.Sum(x => x.DG_DI_SolverCount)));
            int dg_di_verifier = Convert.ToInt32(Math.Round(records.Sum(x => x.DG_DI_VerCount)));

            string file = directory + Path.DirectorySeparatorChar + "report" +
                Path.DirectorySeparatorChar + "tables" + Path.DirectorySeparatorChar + filename;

            string content = File.ReadAllText(file);
            content = content.Replace("@@autosync_time@@", Math.Round(autosync_time, 2).ToString());
            content = content.Replace("@@autosync_median@@", Math.Round(autosync_median, 2).ToString());
            content = content.Replace("@@autosync_verifier@@", autosync_verifier.ToString());

            content = content.Replace("@@default_time@@", Math.Round(default_time, 2).ToString());
            content = content.Replace("@@default_median@@", Math.Round(default_median, 2).ToString());
            content = content.Replace("@@default_solver@@", default_solver.ToString());
            content = content.Replace("@@default_verifier@@", default_verifier.ToString());

            content = content.Replace("@@dg_time@@", Math.Round(dg_time, 2).ToString());
            content = content.Replace("@@dg_median@@", Math.Round(dg_median, 2).ToString());
            content = content.Replace("@@dg_solver@@", dg_solver.ToString());
            content = content.Replace("@@dg_verifier@@", dg_verifier.ToString());

            content = content.Replace("@@di_time@@", Math.Round(di_time, 2).ToString());
            content = content.Replace("@@di_median@@", Math.Round(di_median, 2).ToString());
            content = content.Replace("@@di_solver@@", di_solver.ToString());
            content = content.Replace("@@di_verifier@@", di_verifier.ToString());

            content = content.Replace("@@dg_di_time@@", Math.Round(dg_di_time, 2).ToString());
            content = content.Replace("@@dg_di_median@@", Math.Round(dg_di_median, 2).ToString());
            content = content.Replace("@@dg_di_solver@@", dg_di_solver.ToString());
            content = content.Replace("@@dg_di_verifier@@", dg_di_verifier.ToString());
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
