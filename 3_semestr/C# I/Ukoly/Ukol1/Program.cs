using System.Globalization;
using System.IO.Pipes;


namespace Ukol1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("cs-CZ");
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            //string data = File.ReadAllText(args[0]);
            string data = File.ReadAllText(@"C:\\Users\\admin\\Documents\\GitHub\\VSB\\3_semestr\\C# I\\Ukoly\\Ukol1\\data.txt");

            Product[] products = ParseData(data);

            double sum = GetTotalProductsPrice(products);
            double averageWeight = GetAverageItemWeight(products);

            Console.WriteLine("Produkty:");
            foreach (var produkt in products)
            {
                if (produkt.Count == null)
                {
                    Console.WriteLine(produkt.Name + ": " + "neznámé množství; " + produkt.ItemPrice + " Kč " + produkt.Weight.GetNormalizedValue() + " kg");
                }
                else
                {
                    Console.WriteLine(produkt.Name + ": " + produkt.Count + " ks; " + produkt.ItemPrice + " Kč " + produkt.Weight.GetNormalizedValue() * produkt.Count.Value );
                }

            }
            Console.WriteLine();
            Console.WriteLine("Celková cena produktů: " + sum + " Kč");
            Console.WriteLine("Průměrná váha položky: " + averageWeight + " kg");

        }

        public static double GetTotalProductsPrice(Product[] products)
        {
            double total = 0;
            foreach (var product in products)
            {
                int count = product.Count ?? 0;
                total += product.ItemPrice * count;
            }
            return total;
        }

        public static double GetAverageItemWeight(Product[] products)
        {
            double totalWeight = 0;
            int totalCount = 0;

            foreach (var product in products)
            { 
                double productWeight = product.Weight.GetNormalizedValue(); 

                totalWeight += productWeight;
                totalCount ++;

            }
            return Math.Round(totalWeight / totalCount, 3);
        }



        public static Product[] ParseData(string data)
        {
            List<Product> products = new List<Product>();
            data = data.Trim(' ');
            int index = 0;
            string[] lines = data.Split("\r\n");
            Product currentProduct = new Product();

            foreach (string line in lines) 
            {
                string test = line.Trim();
                Console.WriteLine(test);
                if (test == "products:")
                {
                    continue;  // Tento řádek přeskočí a pokračuje dalším řádkem
                }

                if (!string.IsNullOrEmpty(test))
                {
                    if (test.StartsWith("- price:"))
                    {
                        index = line.IndexOf("price: ");
                        if (index != -1)
                        {
                            //Console.WriteLine(test.Substring(7).Split(":")[1].Trim());
                            double price;
                            if (double.TryParse(test.Substring(7).Split(":")[1].Trim().Replace(".", ","), out price))
                            {
                                currentProduct.ItemPrice = price;
                            }
                        }
                    }

                    else if (test.StartsWith("- quantity:"))
                    {
                        index = line.IndexOf("quantity:");
                        if (index != -1)
                        {
                            //Console.WriteLine(test.Substring(10).Split(":")[1].Trim());
                            int quantity;
                            if (int.TryParse(test.Substring(10).Split(":")[1].Trim(), out quantity))
                            {
                                currentProduct.Count = quantity;
                            }
                        }

                    }

                    else if (test.StartsWith("- weight:"))
                    {
                        index = line.IndexOf("weight:");
                        if (index != -1)
                        {
                            string[] weightParts = test.Substring(7).Split(":")[1].Split(' ');
                            double weightValue;
                            if (double.TryParse(weightParts[1].Replace(".", ","),  out weightValue))
                            {
                                currentProduct.Weight = new Weight
                                {
                                    Hodnota = weightValue,
                                    Jednotka = weightParts[2].Trim() == "g" ? Jednotka.Gram : (weightParts[2].Trim() == "dkg" ? Jednotka.Dekagram: Jednotka.Kilogram)
                                };
                            }
                            //Console.Write(test.Substring(7).Split(":")[1].Split(' ')[1].Trim());
                            //Console.Write(' ');
                            //Console.Write(test.Substring(7).Split(":")[1].Split(' ')[2].Trim());
                            //Console.WriteLine();
                        }
                    }
                    else
                    {
                        // Pokud se objeví nová položka, přidáme předchozí produkt do seznamu
                        if (currentProduct.Name != null)
                        {
                            products.Add(currentProduct);
                        }

                        // Vytvoříme nový produkt pro aktuální záznam
                        currentProduct = new Product();
                        currentProduct.Name = test.Split("-")[1].Split(":")[0].Trim();
                    }
                }
            }
            // Přidání posledního produktu po skončení cyklu
            if (currentProduct.Name != null)
            {
                products.Add(currentProduct);
            }

            return products.ToArray();
        }
    }
}
