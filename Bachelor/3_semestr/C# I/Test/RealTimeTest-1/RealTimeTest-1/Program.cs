using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace RealTimeTest_1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            string path = "data.json";

            // TODO: kód pro načtení dat
            if (!File.Exists(path))
            {
                Console.WriteLine("Soubor 'data.json' nebyl nalezen.");
                return;
            }

            string json = File.ReadAllText(path);

            var JsonData = DeserializeFromJson(json);

            Console.ReadKey();
            //Places places = new Places();

            //foreach (var element in jsonData.Elements)
            //{
            //    var place = element.ToPlace();
            //    if (place != null)
            //    {
            //        places.Add(place);
            //    }
            //}

            //places.Sort();

            //// TODO: volání metody Filter pro získání všech vrcholů.
            //// var peaks = ....

            //foreach (var peak in peaks)
            //{
            //    Console.WriteLine(peak);
            //}
            //places.Save("places.txt");
        }

        private static object DeserializeFromJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ArgumentException("Input JSON cannot be null or empty.");
            }

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true
            };

            return JsonSerializer.Deserialize<List<Uzel>>(json, options);
        }
    }
}




