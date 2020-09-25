namespace GPURepair.ReportGenerator
{
    using System;
    using System.Threading.Tasks;

    public class Program
    {
        public static async Task Main(string[] args)
        {
            if (args.Length == 0)
            {
                Console.WriteLine("Enter the report to generate!");
                return;
            }
            else if (args[0] == "metrics")
            {
                if (args.Length == 1)
                {
                    Console.WriteLine("Specify the directory to search recursively for the metric files!");
                    return;
                }
                else if (args.Length == 2)
                {
                    Console.WriteLine("Specify the filename of the file containing the timings!");
                    return;
                }
                else if (args.Length == 3)
                {
                    Console.WriteLine("Specify the filename of the summary file!");
                    return;
                }

                MetricsGenerator metrics = new MetricsGenerator(args[1], args[2], args[3]);
                await metrics.Generate().ConfigureAwait(false);
            }
            else if (args[0] == "summary")
            {
                if (args.Length == 1)
                {
                    Console.WriteLine("Specify the directory to search recursively for the csv files!");
                    return;
                }

                SummaryGenerator summary = new SummaryGenerator(args[1]);
                await summary.Generate().ConfigureAwait(false);
            }
        }
    }
}
