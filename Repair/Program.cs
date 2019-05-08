﻿using Microsoft.Boogie;
using System;
using System.Linq;

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
            // standard command line options for Boogie
            CommandLineOptions.Install(new CommandLineOptions());
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

            Repairer repairer = new Repairer(filename);
            Microsoft.Boogie.Program program = repairer.Repair();

            using (TokenTextWriter writer = new TokenTextWriter(filename.Replace(".cbpl", ".fixed.cbpl"), true))
                program.Emit(writer);
        }
    }
}