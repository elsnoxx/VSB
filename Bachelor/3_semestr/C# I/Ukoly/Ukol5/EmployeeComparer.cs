using Ukol5;

public class EmployeeComparer : IComparer<Employee>
{
    public int Compare(Employee x, Employee y)
    {
        double avgX = CalculateAverageScore(x);
        double avgY = CalculateAverageScore(y);

        return avgY.CompareTo(avgX);  // Seřazení od nejvyššího průměru
    }

    private double CalculateAverageScore(Employee employee)
    {
        var totalScore = employee.PerformanceScores.Sum(s => s.Score);
        var count = employee.PerformanceScores.Count;
        return count > 0 ? (double)totalScore / count : 0;
    }
}
