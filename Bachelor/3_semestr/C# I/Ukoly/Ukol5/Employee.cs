using System.Text.Json.Serialization;
using System.Xml.Serialization;

namespace Ukol5;

public class Employee
{
    [XmlAttribute("Id")]
    [JsonPropertyName("employeeId")]
    public string ID { get; set; }

    [XmlElement("Name")]
    [JsonPropertyName("name")]
    public string Name { get; set; }

    [XmlElement("Department")]
    [JsonPropertyName("department")]
    public string Department { get; set; }

    [XmlArray("PerformanceScores")]
    [XmlArrayItem("PerformanceScore")]
    [JsonPropertyName("performanceScores")]
    public List<PerformanceScore> PerformanceScores { get; set; } = new List<PerformanceScore>();

    [XmlElement("HireDate")]
    [JsonPropertyName("hireDate")]
    public string HireDate { get; set; }

    public Employee() { }
}
