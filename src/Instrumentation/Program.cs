namespace GPURepair.Instrumentation
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Common;
    using GPURepair.Common.Diagnostics;
    using Microsoft.Boogie;

    public class Program
    {
        /// <summary>
        /// The entry point of the program.
        /// </summary>
        /// <param name="args">The command line arguments.</param>
        public static void Main(string[] args)
        {
            try
            {
                // standard command line options for Boogie
                CommandLineOptions.Install(new GRCommandLineOptions());
                if (!CommandLineOptions.Clo.Parse(args))
                    return;

                CommandLineOptions.Clo.PrintUnstructured = 2;
                if (!CommandLineOptions.Clo.Files.Any())
                    throw new Exception("An input file must be provided!");
                else if (CommandLineOptions.Clo.Files.Count > 1)
                    throw new Exception("GPURepair can work on only one file at a time!");

                Logger.FileName = CommandLineOptions.Clo.Files.First();
                Logger.DetailedLogging = ((GRCommandLineOptions)CommandLineOptions.Clo).DetailedLogging;

                Microsoft.Boogie.Program program;
                Parser.Parse(Logger.FileName, new List<string>() { "FILE_0" }, out program);

                CommandLineOptions.Clo.DoModSetAnalysis = true;
                CommandLineOptions.Clo.PruneInfeasibleEdges = true;

                ResolutionContext context = new ResolutionContext(null);
                program.Resolve(context);

                program.Typecheck();
                new ModSetCollector().DoModSetAnalysis(program);

                // start instrumenting the program
                SourceLanguage sourceLanguage = ((GRCommandLineOptions)CommandLineOptions.Clo).SourceLanguage;
                bool disableInspection = ((GRCommandLineOptions)CommandLineOptions.Clo).DisableInspection;
                bool disableGridBarriers = ((GRCommandLineOptions)CommandLineOptions.Clo).DisableGridBarriers;

                Instrumentor instrumentor = new Instrumentor(program, sourceLanguage,
                    disableInspection, disableGridBarriers);
                instrumentor.Instrument();

                // create the instrumented Boogie IR for the next steps
                using (TokenTextWriter writer = new TokenTextWriter(Logger.FileName.Replace(".gbpl", ".repair.gbpl"), true))
                    program.Emit(writer);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.Message);
                Environment.Exit(200);
            }
        }
    }
}
