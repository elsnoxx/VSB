namespace Ukol5;

public class EmployeeComparer : IComparer<Employee>
{
    public int Compare(Employee x, Employee y)
    {
        double avgX = CalculateAverageScore(x);
        double avgY = CalculateAverageScore(y);

        return avgY.CompareTo(avgX);  // Seřazení od největšího po nejmenší
    }

    private double CalculateAverageScore(Employee employee)
    {
        var totalScore = 0;
        var count = 0;

        foreach (var score in employee.PerformanceScores)
        {
            totalScore += score.Score;
            count++;
        }

        return count > 0 ? (double)totalScore / count : 0;
    }
}
