﻿namespace GPURepair.Repair
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using GPURepair.Common.Diagnostics;
    using GPURepair.Repair.Diagnostics;
    using GPURepair.Repair.Exceptions;
    using GPURepair.Repair.Metadata;
    using Microsoft.Boogie;

    public class Program
    {
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

                Logger.FileName = CommandLineOptions.Clo.Files.First();
                Logger.DetailedLogging = ((GRCommandLineOptions)CommandLineOptions.Clo).DetailedLogging;

                ClauseLogger.FileName = CommandLineOptions.Clo.Files.First();
                ClauseLogger.LogCLauses = ((GRCommandLineOptions)CommandLineOptions.Clo).LogClauses;

                Barrier.LoopDepthWeight = ((GRCommandLineOptions)CommandLineOptions.Clo).LoopDepthWeight;
                Barrier.GridBarrierWeight = ((GRCommandLineOptions)CommandLineOptions.Clo).GridBarrierWeight;

                Dictionary<string, bool> assignments;

                // repair the program
                Solver.SolverType solverType = ((GRCommandLineOptions)CommandLineOptions.Clo).SolverType;
                Repairer.VerificationType verificationType =
                    ((GRCommandLineOptions)CommandLineOptions.Clo).VerificationType;
                bool disableInspection = ((GRCommandLineOptions)CommandLineOptions.Clo).DisableInspection;
                bool useAxioms = ((GRCommandLineOptions)CommandLineOptions.Clo).UseAxioms;

                Microsoft.Boogie.Program program;
                using (Repairer repairer = new Repairer(Logger.FileName, disableInspection, useAxioms))
                    program = repairer.Repair(verificationType, solverType, out assignments);

                SummaryGenerator generator = new SummaryGenerator();
                IEnumerable<string> changes = generator.GenerateSummary(assignments,
                    Logger.FileName.Replace(".cbpl", ".summary"));

                Logger.Log($"Changes;{changes.Count()}");
                Console.WriteLine("Number of changes required: {0}.", changes.Count());

                if (changes.Any())
                {
                    using (TokenTextWriter writer = new TokenTextWriter(Logger.FileName.Replace(".cbpl", ".fixed.cbpl"), true))
                        program.Emit(writer);
                }
            }
            catch (Exception ex)
            {
                Logger.Log($"ExceptionMessage;{ex.Message}");
                Console.Error.WriteLine(ex.Message);

                if (ex is AssertionException)
                    statusCode = 201;
                else if (ex is RepairException)
                    statusCode = 202;
                else if (ex is NonBarrierException)
                    statusCode = 203;
                else if (ex is SummaryGeneratorException)
                    statusCode = 204;
            }

            if (statusCode != null)
                Environment.Exit(statusCode.Value);
        }
    }
}
