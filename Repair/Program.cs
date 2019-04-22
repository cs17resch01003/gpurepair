using Microsoft.Boogie;

namespace GPURepair.Repair
{
    public class Program
    {
        /// <summary>
        /// Hardcoded file name for testing purposes, has to be changed to use the command line argumnent.
        /// </summary>
        private static string filePath = @"D:\Projects\GPURepair\kernel.cbpl";

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

            Repairer repairer = new Repairer(filePath);
            Microsoft.Boogie.Program program = repairer.Repair();

            using (TokenTextWriter writer = new TokenTextWriter(filePath + ".fixed", true))
                program.Emit(writer);
        }
    }
}