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
        public Product(string name, double itemPrice, int? count, Weight weight)
        {
            Name = name;
            ItemPrice = itemPrice;
            Count = count;
            Weight = weight;
        }
    }

    // Struktura pro váhu
    internal struct Weight
    {
        public double Hodnota { get; set; }
        public Jednotka Jednotka { get; set; }

        // Metoda pro převedení váhy na kilogramy
        public double GetNormalizedValue()
        {
            if (Jednotka == Jednotka.Gram)
                return Hodnota / 1000;
            return Hodnota;  // Je už v kilogramech
        }
    }

    // Enum pro jednotky váhy
    internal enum Jednotka
    {
        Gram,
        Kilogram
    }
}
