namespace GPURepair.ReportGenerator
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using System.Threading.Tasks;
    using CsvHelper;

    public static class CsvWrapper
    {
        public static List<T> Read<T>(string file, bool header = true)
        {
            using (StreamReader reader = new StreamReader(file))
            using (CsvReader csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                csv.Configuration.HasHeaderRecord = header;
                csv.Configuration.ReadingExceptionOccurred = (ex) =>
                {
                    Console.WriteLine(ex.Message);
                    return false;
                };

                return csv.GetRecords<T>().ToList();
            }
        }

        public static async Task Write<T>(IEnumerable<T> records, string file)
        {
            FileInfo info = new FileInfo(file);
            if (!info.Directory.Exists)
                info.Directory.Create();

            using (StreamWriter writer = new StreamWriter(file))
            using (CsvWriter csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.Configuration.ShouldQuote = (field, context) =>
                {
                    return true;
                };

                await csv.WriteRecordsAsync(records).ConfigureAwait(false);
            }
        }
    }
}
