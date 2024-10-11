using System.Diagnostics;
using System.Globalization;
using System.IO.Pipes;
using System.Runtime.Intrinsics.X86;
using System.Text.RegularExpressions;


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
                    Console.WriteLine(produkt.Name + ": " + "neznámé množství; " + produkt.ItemPrice + " Kč");
                }
                else
                {
                    Console.WriteLine(produkt.Name + ": " + produkt.Count + " ks; " + produkt.ItemPrice + " Kč");
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
                if (test == "products:" || string.IsNullOrEmpty(test))
                {
                    continue;
                }


                if (test.StartsWith("- price:"))
                {
                    index = line.IndexOf("price: ");
                    if (index != -1)
                    {
                        Regex priceRegex = new Regex(@"- price:\s*(\d+(\.\d+)?)");
                        Match match = priceRegex.Match(line);
                        if (match.Success)
                        {
                            currentProduct.ItemPrice = double.Parse(match.Groups[1].Replace(".", ",").Value);
                        }

                    }
                }

                else if (test.StartsWith("- quantity:"))
                {
                    index = line.IndexOf("quantity:");
                    if (index != -1)
                    {
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
                    }
                }
                else
                {
                    if (currentProduct.Name != null)
                    {
                        products.Add(currentProduct);
                    }
                    currentProduct = new Product();
                    currentProduct.Name = test.Split("-")[1].Split(":")[0].Trim();
                }
            }
            
            if (currentProduct.Name != null)
            {
                products.Add(currentProduct);
            }

            return products.ToArray();
        }
    }
}
