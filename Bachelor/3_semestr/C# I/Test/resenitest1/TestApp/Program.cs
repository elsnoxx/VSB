using System.Globalization;
using System.Text.Json;

namespace TestApp
{
	internal class Program
	{
		static void Main(string[] args)
		{
			Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;


			string path = "data.json";

			string data = File.ReadAllText(path);
			JsonData json = JsonSerializer.Deserialize<JsonData>(data, new JsonSerializerOptions()
			{
				PropertyNameCaseInsensitive = true
			});

			Places places = new Places();


			foreach (Element element in json.Elements)
			{
				var place = element.ToPlace();
				if (place != null)
				{
					places.Add(place);
				}

			}

			places.Sort();

			foreach (var place in places.Filter<Peak>())
			{
				Console.WriteLine(place);
			}

			places.Save("places.txt");
		}
	}
}