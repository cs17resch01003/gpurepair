using GPURepair.Repair.Exceptions;
using Microsoft.Boogie;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static GPURepair.Repair.SummaryGenerator;

namespace GPURepair.Repair
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
                CommandLineOptions.Install(new GRCommandLineOptions());
                if (!CommandLineOptions.Clo.Parse(args))
                    return;

                CommandLineOptions.Clo.PrintUnstructured = 2;
                CommandLineOptions.Clo.DoModSetAnalysis = true;
                CommandLineOptions.Clo.PruneInfeasibleEdges = true;

                if (!CommandLineOptions.Clo.Files.Any())
                    throw new Exception("An input file must be provided!");
                else if (CommandLineOptions.Clo.Files.Count > 1)
                    throw new Exception("GPURepair can work on only one file at a time!");

                string filename = CommandLineOptions.Clo.Files.First();
                Watch.LogEvent += (milliseconds) =>
                {
                    string logFile = ((GRCommandLineOptions)CommandLineOptions.Clo).TimeLog;
                    if (logFile != null)
                        File.AppendAllLines(logFile, new string[] { string.Format("{0},{1}", filename, milliseconds) });
                };

                Dictionary<string, bool> assignments;

                Repairer repairer = new Repairer(filename);
                Microsoft.Boogie.Program program = repairer.Repair(out assignments);

                SummaryGenerator generator = new SummaryGenerator(program, assignments);
                IEnumerable<Location> changes = generator.GenerateSummary(filename.Replace(".cbpl", ".summary"));

                Console.WriteLine("Number of changes required: {0}.", changes.Count());
                if (changes.Any())
                {
                    using (TokenTextWriter writer = new TokenTextWriter(filename.Replace(".cbpl", ".fixed.cbpl"), true))
                        program.Emit(writer);
                }

                foreach (Location location in changes)
                    if (new FileInfo(location.File).FullName != new FileInfo(filename).FullName)
                        throw new RepairError("There are changes needed in external files: " + location.ToString());
            }
            catch (AssertionError ex)
            {
                Console.Error.WriteLine(ex.Message);
                Environment.Exit(201);
            }
            catch (RepairError ex)
            {
                Console.Error.WriteLine(ex.Message);
                Environment.Exit(202);
            }
        }
    }
}