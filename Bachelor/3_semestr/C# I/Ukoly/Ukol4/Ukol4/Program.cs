using System.Collections.Generic;
using System.IO;


namespace Ukol4
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ObservableList<Customer> list = new ObservableList<Customer>();

            // Připojení k událostem
            list.OnAdd += item => Console.WriteLine($"Přidáno: {item}");
            list.OnRemove += item => Console.WriteLine($"Odebráno: {item}");

            // LoadFromBin(list, @"C:\Users\ficek\OneDrive\Dokumenty\GitHub\VSB\3_semestr\C# I\Ukoly\Ukol4\Ukol4\CSI_2024W_JAW254_du4\test0\source1.bin");
            LoadFromBin(list, args[0]);
            
            list.Remove(list[2]);

            string json = list.SerializeToJson();
            string xml = list.SerializeToXml();

            // Tento řádek je nutný kvůli správnému fungování v Kelvinu!
            xml = xml.Trim(new char[] { '\uFEFF', '\u200B' });

            SaveToTextFile(json, "customers.json");
            SaveToTextFile(xml, "customers.xml");

            Console.WriteLine(new string('\n', 5));

            Console.WriteLine(new string('=', 20));
            Console.WriteLine("JSON data");
            Console.WriteLine(new string('=', 20));
            Console.WriteLine();

            ObservableList<Customer> jsonCustomers = ObservableList<Customer>.DeserializeFromJson(
                LoadFromTextFile("customers.json")
                );
            Print(jsonCustomers);

            Console.WriteLine(new string('\n', 5));

            Console.WriteLine(new string('=', 20));
            Console.WriteLine("XML data");
            Console.WriteLine(new string('=', 20));
            Console.WriteLine();

            ObservableList<Customer> xmlCustomers = ObservableList<Customer>.DeserializeFromXml(
                LoadFromTextFile("customers.xml")
                );
            Print(xmlCustomers);
            // Console.ReadKey();
        }
        private static void Print(ObservableList<Customer> customers)
        {
            Console.WriteLine("Zákazníci:");
            Console.WriteLine(new string('-', 15));
            foreach (Customer num in customers)
            {
                Console.WriteLine(num);
            }


            Console.WriteLine("\n");
            Console.WriteLine("Nezletilí zákazníci:");
            Console.WriteLine(new string('-', 15));
            foreach (Customer num in customers.Filter(
                /*
                Zde bude možné předat libovolnou metodu přijímající Customer a vracející bool.
                Pro otestování zde doplňte metodu viz. zadání.
                */
                // TODO: doplnit...
                c => c.Age < 18
                ))
            {
                Console.WriteLine(num);
            }


            Console.WriteLine("\n");
            Console.WriteLine("Zákazníci seřazení podle věku:");
            Console.WriteLine(new string('-', 15));
            foreach (Customer num in customers.OrderBy(new AgeComparer()))
            {
                Console.WriteLine(num);
            }
        }

        static void LoadFromBin(ObservableList<Customer> list, string path)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(path))
                {
                    throw new ArgumentException("Path cannot be null or empty.");
                }

                if (!File.Exists(path))
                {
                    throw new FileNotFoundException($"File not found at the specified path: {path}");
                }

                using (var stream = new FileStream(path, FileMode.Open))
                using (var reader = new BinaryReader(stream))
                {
                    int recordCount = reader.ReadInt32();
                    for (int i = 0; i < recordCount; i++)
                    {
                        string name = reader.ReadString();
                        int age = reader.ReadInt32();

                        var customer = new Customer(name, age);
                        list.Add(customer);
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
        }

        public static string LoadFromTextFile(string filePath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(filePath))
                {
                    throw new ArgumentException("File path cannot be null or empty.");
                }

                if (!File.Exists(filePath))
                {
                    throw new FileNotFoundException($"File not found at the specified path: {filePath}");
                }

                return File.ReadAllText(filePath);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
                return string.Empty;
            }
        }

        public static void SaveToTextFile(string content, string filePath)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(content))
                {
                    throw new ArgumentException("Content cannot be null or empty.");
                }

                if (string.IsNullOrWhiteSpace(filePath))
                {
                    throw new ArgumentException("File path cannot be null or empty.");
                }

                File.WriteAllText(filePath, content);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
        }
    }
}
