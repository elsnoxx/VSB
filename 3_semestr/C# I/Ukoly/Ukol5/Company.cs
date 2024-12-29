using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;


namespace Ukol5;

public class Company
{
    public List<Employee> Employees = new List<Employee>();
    
    public Company(){}
    
    
    public static Company DeserializeFromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
        {
            throw new ArgumentException("Input JSON cannot be null or empty.");
        }

        var options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true 
        };

        // Deserializace na List<Employee>, ne List<T>
        var deserializedItems = JsonSerializer.Deserialize<List<Employee>>(json, options);
        var company = new Company();

        if (deserializedItems != null)
        {
            foreach (var item in deserializedItems)
            {
                company.Employees.Add(item);  // Přidání zaměstnanců do firmy
            }
        }

        return company;  // Vracení instance Company
    }



    public void PrintAvegateScoreByMonthOfYear()
    {
        // Soubory pro uchování součtů a počtů skóre za jednotlivé měsíce
        var monthScores = new Dictionary<string, (int sum, int count)>();

        foreach (var employee in Employees)
        {
            foreach (var score in employee.PerformanceScores)
            {
                if (!monthScores.ContainsKey(score.Month))
                {
                    monthScores[score.Month] = (0, 0);
                }

                monthScores[score.Month] = (monthScores[score.Month].sum + score.Score, monthScores[score.Month].count + 1);
            }
        }

        // Výpočet a výpis průměrů pro jednotlivé měsíce
        foreach (var month in monthScores)
        {
            var average = month.Value.sum / (double)month.Value.count;
            Console.WriteLine($"{month.Key}: {average:F1}");  // Formátování průměru na jedno desetinné místo
        }
    }

    public void GetEmployeeScores()
    {
        
    }
    
    
    public void Save(string fileName)
    {
        var serializer = new XmlSerializer(typeof(List<Employee>));
        using (var writer = new StreamWriter(fileName))
        {
            serializer.Serialize(writer, Employees);
        }
    }


    
}