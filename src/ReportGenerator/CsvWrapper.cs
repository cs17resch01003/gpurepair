namespace GPURepair.ReportGenerator
{
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using CsvHelper;

    public static class CsvWrapper
    {
        public static List<string> ReadLines(string file)
        {
            return File.ReadAllLines(file).ToList();
        }

        public static List<T> Read<T>(string file, bool header = true)
        {
            using (StreamReader reader = new StreamReader(file))
            using (CsvReader csv = new CsvReader(reader))
            {
                csv.Configuration.HasHeaderRecord = header;
                csv.Configuration.MissingFieldFound = null;

                return csv.GetRecords<T>().ToList();
            }
        }

        public static void Write<T>(IEnumerable<T> records, string file)
        {
            FileInfo info = new FileInfo(file);
            if (!info.Directory.Exists)
                info.Directory.Create();

            using (StreamWriter writer = new StreamWriter(file))
            using (CsvWriter csv = new CsvWriter(writer))
            {
                csv.Configuration.QuoteAllFields = true;
                csv.WriteRecords(records);
            }
        }
    }
}
