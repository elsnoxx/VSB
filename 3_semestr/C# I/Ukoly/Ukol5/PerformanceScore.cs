namespace Ukol5;

public class PerformanceScore
{
    public string Month { get; set; }
    public int Score { get; set; }
    
    public PerformanceScore() { }

    public PerformanceScore(string month, int score)
    {
        this.Month = month;
        this.Score = score;
    }

    public override string ToString()
    {
        return $"{Month} - {Score}";
    }

}