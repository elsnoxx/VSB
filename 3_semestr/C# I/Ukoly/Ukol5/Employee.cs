using System.Xml.Serialization;

namespace Ukol5;

public class Employee
{
    [XmlAttribute("Id")]
    public string ID { get; set; }

    [XmlElement("Name")]
    public string Name { get; set; }

    [XmlElement("Department")]
    public string Department { get; set; }

    [XmlArray("PerformanceScores")]
    [XmlArrayItem("PerformanceScore")]
    public List<PerformanceScore> PerformanceScores { get; set; } = new List<PerformanceScore>();

    [XmlElement("HireDate")]
    public string HireDate { get; set; }

    public Employee() { }
}
