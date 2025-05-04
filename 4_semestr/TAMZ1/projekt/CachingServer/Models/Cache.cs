using System.Text.Json.Serialization;

namespace CachingServer.Models
{
    public class Cache
    {
        [JsonPropertyName("id")]
        public int Id { get; set; }

        [JsonPropertyName("name")]
        public string? Name { get; set; }

        [JsonPropertyName("description")]
        public string? Description { get; set; }

        [JsonPropertyName("lat")]
        public double Lat { get; set; }

        [JsonPropertyName("lng")]
        public double Lng { get; set; }

        [JsonPropertyName("date")]
        public DateTime? Date { get; set; }

        [JsonPropertyName("owner")]
        public string? Owner { get; set; }

        [JsonPropertyName("difficulty")]
        public double Difficulty { get; set; }

        [JsonPropertyName("size")]
        public string? Size { get; set; }

        [JsonPropertyName("foundCount")]
        public int FoundCount { get; set; }

        [JsonPropertyName("hint")]
        public string? Hint { get; set; }

        [JsonPropertyName("logs")]
        public List<Log>? Logs { get; set; }
    }

    public class Log
    {
        [JsonPropertyName("user")]
        public string? User { get; set; }

        [JsonPropertyName("date")]
        public DateTime? Date { get; set; }

        [JsonPropertyName("text")]
        public string? Text { get; set; }
    }
}
