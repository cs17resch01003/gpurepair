namespace GPURepair.ReportGenerator
{
    public class SummaryGenerator
    {
        private string directory;

        public SummaryGenerator(string directory)
        {
            this.directory = directory;
        }

        public void Generate()
        {
            FileParser.ParseFiles(directory);
            DataAnalyzer.AnalyzeData(directory);
            TexGenerator.Generate(directory);
        }
    }
}
