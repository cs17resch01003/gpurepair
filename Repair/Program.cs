using GPURepair.Repair.Exceptions;
using GPURepair.Solvers;
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
        private static string logFile;

        private static string filename;

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

                filename = CommandLineOptions.Clo.Files.First();
                logFile = ((GRCommandLineOptions)CommandLineOptions.Clo).RepairLog;

                Dictionary<string, bool> assignments;

                Repairer repairer = new Repairer(filename);
                Microsoft.Boogie.Program program = repairer.Repair(out assignments, ((GRCommandLineOptions)CommandLineOptions.Clo).EnableEfficientSolving);

                SummaryGenerator generator = new SummaryGenerator(program, assignments);
                IEnumerable<Location> changes = generator.GenerateSummary(filename.Replace(".cbpl", ".summary"));

                Console.WriteLine("Number of changes required: {0}.", changes.Count());
                if (changes.Any())
                {
                    using (TokenTextWriter writer = new TokenTextWriter(filename.Replace(".cbpl", ".fixed.cbpl"), true))
                        program.Emit(writer);
                }

                string cu_source = new FileInfo(filename).FullName.Replace(".cbpl", ".cu");
                string cl_source = new FileInfo(filename).FullName.Replace(".cbpl", ".cl");

                foreach (Location location in changes)
                {
                    if (cu_source != location.File && cl_source != location.File)
                        throw new RepairError("There are changes needed in external files: " + location.ToString());
                }

                Logger.Log(logFile, filename);
            }
            catch (AssertionError ex)
            {
                Logger.Log(logFile, filename);

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(201);
            }
            catch (RepairError ex)
            {
                Logger.Log(logFile, filename);

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(202);
            }
            catch (NonBarrierError ex)
            {
                Logger.Log(logFile, filename);

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(203);
            }
            catch (SummaryGeneratorError ex)
            {
                Logger.Log(logFile, filename);

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(204);
            }
            catch (Exception ex)
            {
                Logger.Log(logFile, filename);

                Console.Error.WriteLine(ex.Message);
                Environment.Exit(-1);
            }
        }
    }
}