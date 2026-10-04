using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opravatestuRealtime
{
    public class AttractionList : IEnumerable
    {
        public List<Attraction> Attractions { get; set; }

        
        public AttractionList()
        {
            Attractions = new List<Attraction>();
        }


        public IEnumerator GetEnumerable()
        {
            int timeNow = DateTime.Now.Hour;
            foreach (var item in Attractions)
            {
                if (item is IOpeningHours openingHours)
                {
                    if (openingHours.OpenTime.HasValue && openingHours.CloseTime.HasValue)
                    {
                        if (timeNow >= openingHours.OpenTime && timeNow <= openingHours.CloseTime)
                        {
                            yield return item;
                        }
                    }
                }
            }
        }
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerable();

        public void Load(string path)
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
                        int type = reader.ReadInt32();
                        if (type == 0)
                        {

                            string name = reader.ReadString();
                            int? openTime = null;
                            int? closeTime = null;
                            if (reader.ReadBoolean())
                            {
                                openTime = reader.ReadInt32();
                            }
                            if (reader.ReadBoolean())
                            {
                                closeTime = reader.ReadInt32();
                            }

                            Attractions.Add(new Casle() { Name = name, OpenTime = openTime, CloseTime = closeTime });
                            
                        }
                        else if (type == 1)
                        {
                            string name = reader.ReadString();
                            int? openTime = null;
                            int? closeTime = null;
                            if (reader.ReadBoolean())
                            {
                                openTime = reader.ReadInt32();
                            }
                            if (reader.ReadBoolean())
                            {
                                closeTime = reader.ReadInt32();
                            }

                            Attractions.Add(new Aquapark() { Name = name, OpenTime = openTime, CloseTime = closeTime });
                        }
                        else if (type == 2)
                        {
                            string name = reader.ReadString();
                            double length = reader.ReadDouble();
                            Attractions.Add(new Bridge() { Name = name, Length = length });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"An unexpected error occurred: {ex.Message}");
            }
        }

        public void Save(string file_path)
        {
            
            using (var stream = new FileStream(file_path, FileMode.Create))
            using (var sw = new StreamWriter(stream))
            {
                foreach (var item in Attractions)
                {
                    sw.WriteLine(item);
                }
            }
        }


    }
}
