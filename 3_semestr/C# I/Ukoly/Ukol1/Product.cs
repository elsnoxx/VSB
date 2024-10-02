namespace Ukol1
{
    internal class Product
    {
        // Vlastnosti produktu
        public string Name { get; set; }
        public double ItemPrice { get; set; }
        public int? Count { get; set; }  // Nullable, protože počet může být neznámý
        public Weight Weight { get; set; }

        // Konstruktor
        public Product() { }
    }

    // Struktura pro váhu
    internal struct Weight
    {
        public double Hodnota { get; set; }
        public Jednotka Jednotka { get; set; }

        // Metoda pro převedení váhy na kilogramy
        public double GetNormalizedValue()
        {
            // Normalizace hmotnosti na kilogramy
            switch (Jednotka)
            {
                case Jednotka.Gram:
                    return Hodnota / 1000;
                case Jednotka.Dekagram:
                    return Hodnota / 100;
                case Jednotka.Kilogram:
                    return Hodnota;
                default:
                    return 0;
            }
        }
    }

    // Enum pro jednotky váhy
    internal enum Jednotka
    {
        Gram,
        Kilogram,
        Dekagram
    }
}
