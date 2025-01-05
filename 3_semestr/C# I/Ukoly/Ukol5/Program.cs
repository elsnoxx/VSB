using System.Globalization;
using System.IO;
using System.Text.Json;

namespace Ukol5
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Nastavení kulturního prostředí
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;

            // Načtení JSON souboru
            //string jsonFilePath = "C:\\Users\\ficek\\Documents\\GitHub\\VSB\\3_semestr\\C# I\\Ukoly\\Ukol5\\DataIn\\data.json";
            string jsonFilePath = "data.json";
            if (!File.Exists(jsonFilePath))
            {
                Console.WriteLine("Soubor 'data.json' nebyl nalezen.");
                return;
            }

            string json = File.ReadAllText(jsonFilePath);

            // Deserializace JSON na seznam zaměstnanců
            var employees = DeserializeFromJson(json);

            // Vytvoření objektu Company
            Company company = new Company();

            // Naplnění společnosti zaměstnanci
            foreach (Employee employee in employees)
            {
                company.Add(employee);
            }

            Console.WriteLine("Průměrné skóre podle měsíců:");
            Console.WriteLine("---------------");
            company.PrintAvegateScoreByMonthOfYear();

            Console.WriteLine();

            // Seřazení zaměstnanců podle průměrného skóre
            company.Sort(new EmployeeComparer());

            // Enumerování skóry zaměstnanců
            Console.WriteLine("Zaměstnanci a jejich skóre v jednotlivých měsících:");
            Console.WriteLine("---------------");
            foreach (var score in company.GetEmployeeScores())
            {
                Console.WriteLine($"{score.EmployeeName}: {score.Month} - {score.Score}");
            }

            // Serializace do XML
            company.Save("data.xml");
        }

        public static List<Employee> DeserializeFromJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ArgumentException("Input JSON cannot be null or empty.");
            }

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            return JsonSerializer.Deserialize<List<Employee>>(json, options);
        }

    }
}
