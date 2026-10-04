using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;
using System.Xml;
using System.Xml.Serialization;


namespace Ukol5;

[XmlRoot("ArrayOfEmployee")]
public class Company
{
    [XmlElement("Employee", Type = typeof(Employee))]
    public List<Employee> Employees { get; set; } = new List<Employee>();

    public Company(){}


    public void PrintAvegateScoreByMonthOfYear()
    {
        var monthScores = new Dictionary<string, (int sum, int count)>();

        foreach (var employee in Employees)
        {
            foreach (var score in employee.PerformanceScores)
            {
                if (!monthScores.ContainsKey(score.Month))
                {
                    monthScores[score.Month] = (0, 0);
                }

                monthScores[score.Month] = (
                    monthScores[score.Month].sum + score.Score,
                    monthScores[score.Month].count + 1
                );
            }
        }

        foreach (var month in monthScores)
        {
            double average = month.Value.sum / (double)month.Value.count;
            Console.WriteLine($"{month.Key}: {average:F1}");
        }
    }


    public IEnumerable<EmployeeScore> GetEmployeeScores()
    {
        foreach (var employee in Employees)
        {
            foreach (var score in employee.PerformanceScores)
            {
                yield return new EmployeeScore(employee.Name, score.Month, score.Score);
            }
        }
    }


    public void Add(Employee employee)
    {
        if (employee == null)
        {
            throw new ArgumentNullException(nameof(employee), "Nelze pøidat null zamìstnance.");
        }

        if (string.IsNullOrWhiteSpace(employee.ID))
        {
            throw new ArgumentException("ID zamìstnance nesmí být prázdné.");
        }

        Employees.Add(employee);
    }

    public void Sort(IComparer<Employee> comparer)
    {
        Employees.Sort(comparer);
    }


    public void Save(string fileName)
    {
        var serializer = new System.Xml.Serialization.XmlSerializer(typeof(Company));

        var settings = new XmlWriterSettings
        {
            Indent = true,
            Encoding = Encoding.UTF8,
            OmitXmlDeclaration = false,
            NewLineChars = "\n"
        };

        using (var writer = XmlWriter.Create(fileName, settings))
        {
            serializer.Serialize(writer, this);
        }
    }
}