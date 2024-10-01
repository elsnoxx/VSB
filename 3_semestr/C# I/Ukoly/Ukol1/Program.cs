using System.Globalization;
using System.Security;

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

            foreach (var produkt in products)
            {
                Console.WriteLine(produkt.Name+ ' ' + produkt.ItemPrice);
            }
        }

        public static Product[] ParseData(string data)
        {
            List<Product> products = new List<Product>();
            data = data.Trim(' ');
            int index = 0;
            string[] lines = data.Split('\n');
            Product currentProduct = null;

            foreach (string line in lines) 
            {
                string test = line.Trim();

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
                            if (double.TryParse(test.Substring(7).Split(":")[1].Trim(), out price))
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
                            Console.Write(test.Substring(7).Split(":")[1].Split(' ')[1].Trim());
                            Console.Write('-');
                            Console.Write(test.Substring(7).Split(":")[1].Split(' ')[2].Trim());
                            Console.WriteLine();
                        }
                    }

                    else
                    {
                        Console.WriteLine(test.Split("-")[1].Split(":")[0].Trim());
                        //currentProduct.Name = test.Split("-")[1].Split(":")[0].Trim();
                    }
                }
            }

            return products.ToArray();
        }
    }
}
