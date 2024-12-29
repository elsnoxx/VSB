using System.Globalization;


namespace Ukol5
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Nastavení kulturní informace
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            CultureInfo.CurrentUICulture = CultureInfo.InvariantCulture;

            // Načtení JSON souboru
            string jsonFilePath = "C:\\Users\\ficek\\OneDrive\\Dokumenty\\GitHub\\VSB\\3_semestr\\C# I\\Ukoly\\Ukol5\\DataIn\\data.json";
            if (File.Exists(jsonFilePath))
            {
                string json = File.ReadAllText(jsonFilePath);

                // Deserializace a naplnění firmy
                Company company = Company.DeserializeFromJson(json);

                Console.WriteLine("Průměrné skóre podle měsíců:");
                Console.WriteLine("---------------");
                company.PrintAvegateScoreByMonthOfYear(); // Výpis průměrného skóre

                Console.WriteLine();

                // Seřazení zaměstnanců podle průměrného skóre
                company.Sort(new EmployeeComparer());

                // Enumerování zaměstnanců a jejich skóre
                Console.WriteLine("Zaměstnanci a jejich skóre v jednotlivých měsících:");
                Console.WriteLine("---------------");
                foreach (EmployeeScore score in company.GetEmployeeScores())
                {
                    Console.WriteLine(score.EmployeeName + ": " + score.Month + " - " + score.Score);
                }

                // Serializace do XML
                company.Save("data.xml");
            }
            else
            {
                Console.WriteLine("Soubor 'data.json' nebyl nalezen.");
            }
            
        }
    }
}
