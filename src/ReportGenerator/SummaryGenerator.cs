namespace GPURepair.ReportGenerator
{
    using System.Threading.Tasks;

    public class SummaryGenerator
    {
        private string directory;

        public SummaryGenerator(string directory)
        {
            this.directory = directory;
        }

        public async Task Generate()
        {
            await FileParser.ParseFiles(directory);
            await DataAnalyzer.AnalyzeData(directory);
            TexGenerator.Generate(directory);
        }
    }
}
