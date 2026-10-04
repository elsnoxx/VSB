using System.Diagnostics;
using System.Globalization;
using System.Numerics;
using System.Text.Json;

namespace PROJEKT
{
    public class Program
    {
        static void Main()
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;

            string path = "data.json";
            string jsonContent = File.ReadAllText(path);

            Root root = JsonSerializer.Deserialize<Root>(jsonContent) ?? new Root();
            List<Element> elements = root.elements;

            Places places = new Places();
            foreach (var element in elements)
            {
                var place = element.ToPlace();
                if (place != null)
                {
                    places.Add(place);
                }
            }

            // Setřídění míst dle zadání
            places.Sort();

            // Získání všech vrcholů
            foreach (var peak in places.Filter<Peak>())
            {
                Console.WriteLine(peak);
            }

            foreach (var shop in places.Filter<Shop>())
            {
                Console.WriteLine(shop);
            }

            // Uložení seznamu míst do souboru
            places.Save("places.txt", places);
        }
    }

    public class Root
    {
        public double version { get; set; }
        public string generator { get; set; }
        public Osm3s osm3s { get; set; }
        public List<Element> elements { get; set; }
    }

    public class Osm3s
    {
        public string timestamp_osm_base { get; set; }
        public string copyright { get; set; }
    }

    public class Element
    {
        public string type { get; set; }
        public long id { get; set; }
        public double lat { get; set; }
        public double lon { get; set; }
        public Dictionary<string, string> tags { get; set; }

        public object ToPlace()
        {
            if (tags.ContainsKey("natural"))
            {
                return new Peak
                {
                    Id = id,
                    Lat = lat,
                    Lon = lon,
                    Name = tags.ContainsKey("name") ? tags["name"] : "Neznámý název",
                    Elevation = tags.ContainsKey("ele") ? double.Parse(tags["ele"]) : 0
                };
            }
            else if (tags.ContainsKey("shop"))
            {
                return new Shop
                {
                    Id = id,
                    Lat = lat,
                    Lon = lon,
                    Name = tags.ContainsKey("name") ? tags["name"] : "Neznámý název",
                    Type = tags["shop"],
                    OpeningHours = tags.ContainsKey("opening_hours") ? tags["opening_hours"] : null
                };
            }
            return null;
        }
    }

    public class Peak
    {
        public long Id { get; set; }
        public double Lat { get; set; }
        public double Lon { get; set; }
        public string Name { get; set; }
        public double Elevation { get; set; }

        public override string ToString()
        {
            if (Elevation == 0)
            {
                return $"VRCHOL | {Name} ({Lat}, {Lon}): ???m.n.m";
            }

            return $"VRCHOL | {Name} ({Lat}, {Lon}): {Elevation}m.n.m";
        }
    }

    public class Shop
    {
        public long Id { get; set; }
        public double Lat { get; set; }
        public double Lon { get; set; }
        public string Name { get; set; }
        public string Type { get; set; }
        public string OpeningHours { get; set; }

        public override string ToString()
        {
            if (OpeningHours == null)
            {
                return $"OBCHOD | {Name} ({Lat}, {Lon}): {Type} - ???";
            }

            return $"OBCHOD | {Name} ({Lat}, {Lon}): {Type} - {OpeningHours}";
        }
    }

    public class Places
    {
        private List<object> _places;

        public Places()
        {
            _places = new List<object>();
        }

        public void Add(object place)
        {
            _places.Add(place);
        }

        public List<object> GetAll()
        {
            return _places;
        }

        public void Sort()
        {
            _places = _places.OrderByDescending(p => p is Peak ? (p as Peak).Elevation : 0)
                             .ThenBy(p => p is Peak ? (p as Peak).Name : string.Empty)
                             .ThenBy(p => p is Shop ? (p as Shop).Name : string.Empty)
                             .ToList();
        }

        public IEnumerable<T> Filter<T>()
        {
            foreach (var place in _places)
            {
                if (place is T)
                {
                    yield return (T)place;
                }
            }
        }

        public void Save(string filePath, Places places)
        {
            // Uložení do textového souboru
            using (var streamWriter = new StreamWriter(filePath))
            {
                // Uložení do textového souboru
                foreach (var peak in places.Filter<Peak>())
                {
                    streamWriter.WriteLine(peak.ToString());
                }
            }
        }
    }
}
