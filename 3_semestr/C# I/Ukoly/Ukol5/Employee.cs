namespace Ukol5;

public class Employee
{
    public int ID { get; set; }
    public string Name { get; set; }
    public string Department { get; set; }
    public List<PerformanceScore> PerformanceScores = new List<PerformanceScore>();
    public string HireDate { get; set; }
    
    public Employee() { }
}