namespace Ukol5;

public class EmployeeScore
{
    public string EmployeeName { get; set; }
    public string Month { get; set; }
    public int Score { get; set; }

    public EmployeeScore(string employeeName, string month, int score)
    {
        EmployeeName = employeeName;
        Month = month;
        Score = score;
    }
}
