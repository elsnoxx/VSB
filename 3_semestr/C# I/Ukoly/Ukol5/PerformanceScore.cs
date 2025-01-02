using System.Xml.Serialization;

namespace Ukol5;

public class PerformanceScore
{
    [XmlElement("Month")]
    public string Month { get; set; }

    [XmlElement("Score")]
    public int Score { get; set; }

    public PerformanceScore() { }

    public PerformanceScore(string month, int score)
    {
        Month = month;
        Score = score;
    }
}
