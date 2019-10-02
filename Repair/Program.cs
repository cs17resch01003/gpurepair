﻿using GPURepair.Repair.Diagnostics;
using GPURepair.Repair.Exceptions;
using Microsoft.Boogie;
using System;
using System.Collections.Generic;
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
            int? statusCode = null;

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

                Logger.FileName = filename = CommandLineOptions.Clo.Files.First();
                Logger.LogFile = logFile = ((GRCommandLineOptions)CommandLineOptions.Clo).RepairLog;
                Logger.LogCLauses = ((GRCommandLineOptions)CommandLineOptions.Clo).LogClauses;

                Dictionary<string, bool> assignments;

                Repairer repairer = new Repairer(filename);
                Microsoft.Boogie.Program program = repairer.Repair(out assignments);

                SummaryGenerator generator = new SummaryGenerator(program, assignments);
                IEnumerable<Location> changes = generator.GenerateSummary(filename.Replace(".cbpl", ".summary"));

                Logger.Changes = changes.Count();
                Console.WriteLine("Number of changes required: {0}.", changes.Count());

                if (changes.Any())
                {
                    using (TokenTextWriter writer = new TokenTextWriter(filename.Replace(".cbpl", ".fixed.cbpl"), true))
                        program.Emit(writer);
                }
            }
            catch (Exception ex)
            {
                Logger.ExceptionMessage = ex.Message;
                Console.Error.WriteLine(ex.Message);

                if (ex is AssertionError)
                    statusCode = 201;
                else if (ex is RepairError)
                    statusCode = 202;
                else if (ex is NonBarrierError)
                    statusCode = 203;
                else if (ex is SummaryGeneratorError)
                    statusCode = 204;
            }
            finally
            {
                Logger.Flush();
            }

            if (statusCode != null)
                Environment.Exit(statusCode.Value);
        }
    }
}