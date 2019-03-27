using Microsoft.Boogie;
using System.Collections.Generic;

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
            // hardcoded file name for testing purposes, has to be changed to use the command line argumnent
            string filename = @"D:\Projects\GPURepair\kernel.gbpl";

            // standard command line options for Boogie
            CommandLineOptions.Install(new CommandLineOptions());
            if (!CommandLineOptions.Clo.Parse(args))
                return;

            CommandLineOptions.Clo.PrintUnstructured = 2;

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
            using (TokenTextWriter writer = new TokenTextWriter(@"D:\Projects\GPURepair\kernel.repair.gbpl", true))
                program.Emit(writer);
        }
    }
}