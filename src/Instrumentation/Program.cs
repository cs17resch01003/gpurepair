using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Boogie;

namespace GPURepair.Instrumentation
{
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
                CommandLineOptions.Install(new CommandLineOptions());
                if (!CommandLineOptions.Clo.Parse(args))
                    return;

                CommandLineOptions.Clo.PrintUnstructured = 2;
                if (!CommandLineOptions.Clo.Files.Any())
                    throw new Exception("An input file must be provided!");
                else if (CommandLineOptions.Clo.Files.Count > 1)
                    throw new Exception("GPURepair can work on only one file at a time!");

                string filename = CommandLineOptions.Clo.Files.First();

                Microsoft.Boogie.Program program;
                Parser.Parse(filename, new List<string>() { "FILE_0" }, out program);

                CommandLineOptions.Clo.DoModSetAnalysis = true;
                CommandLineOptions.Clo.PruneInfeasibleEdges = true;

                ResolutionContext context = new ResolutionContext(null);
                program.Resolve(context);

                program.Typecheck();
                new ModSetCollector().DoModSetAnalysis(program);

                // start instrumenting the program
                Instrumentor instrumentor = new Instrumentor(program);
                instrumentor.Instrument();

                // create the instrumented Boogie IR for the next steps
                // hardcoded file name for testing purposes, has to be changed to use the command line argumnent
                using (TokenTextWriter writer = new TokenTextWriter(filename.Replace(".gbpl", ".repair.gbpl"), true))
                    program.Emit(writer);
            }
            finally
            {
                //Console.Error.WriteLine(ex.Message);
                Environment.Exit(200);
            }
        }
    }
}
