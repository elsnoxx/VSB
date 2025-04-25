namespace apicalls
{
    public class Company
    {
        public string ico { get; set; }
        public string obchodniJmeno { get; set; }
        public string dic { get; set; }
        public Sidlo sidlo { get; set; }
        public string poznamka { get; set; }
    }
    public class Sidlo
    {
        public string nazevObce { get; set; }
    }
}
