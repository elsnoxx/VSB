using System.IO;
using System.Numerics;
using System.Text.Json;
using System.Xml.Linq;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace opravaTest
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
            string path = "data.json";
            string data = File.ReadAllText(path);
            Data json = JsonSerializer.Deserialize<Data>(data, new JsonSerializerOptions()
            {
                PropertyNameCaseInsensitive = true
            });


            foreach (var element in json.Elements)
            {
                Console.WriteLine($"{element}");
            }
        }
    }
}
