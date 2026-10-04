namespace opravaTest
{
    public class Element
    {
        public string Type { get; set; }
        public long Id { get; set; }
        public double Lat { get; set; }
        public double Lon { get; set; }
        public Dictionary<string, string> Tags { get; set; }
    }
}