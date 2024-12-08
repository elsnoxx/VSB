using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Text.Json;

namespace Ukol4
{
    public class ObservableList<T> : IEnumerable<T>
    {
        private List<T> items = new List<T>();

        // Definice delegátů a událostí
        public delegate void ItemAddedHandler(T addedItem);
        public event ItemAddedHandler OnAdd;

        public delegate void ItemRemovedHandler(T removedItem);
        public event ItemRemovedHandler OnRemove;

        public void Add(T item)
        {
            items.Add(item);
            OnAdd?.Invoke(item);
        }

        public void Remove(T item)
        {
            items.Remove(item);
            OnRemove?.Invoke(item);
        }

        public T this[int index]
        {
            get => items[index];
            set => items[index] = value;
        }

        public void PrintAll()
        {
            foreach (var item in items)
            {
                Console.WriteLine(item);
            }
        }

        public string SerializeToJson()
        {
            JsonSerializerOptions options = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase, // Názvy vlastností jako CamelCase
                Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping, // Bez escapování speciálních znaků
                WriteIndented = true // Přehledné odsazení a zalamování řádků
            };

            var json = JsonSerializer.Serialize(items, options);

            return json;
        }

        public string SerializeToXml()
        {
            var serializer = new System.Xml.Serialization.XmlSerializer(typeof(List<T>));

            using (var memoryStream = new MemoryStream())
            using (var writer = new StreamWriter(memoryStream))
            {
                serializer.Serialize(writer, items);
                writer.Flush();
                return Encoding.UTF8.GetString(memoryStream.ToArray());
            }
        }

        public static ObservableList<T> DeserializeFromJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ArgumentException("Input JSON cannot be null or empty.");
            }

            var options = new JsonSerializerOptions
            {
                PropertyNameCaseInsensitive = true 
            };

            var deserializedItems = JsonSerializer.Deserialize<List<T>>(json, options);
            var observableList = new ObservableList<T>();
            if (deserializedItems != null)
            {
                foreach (var item in deserializedItems)
                {
                    observableList.Add(item);
                }
            }

            return observableList;
        }



        public static ObservableList<T> DeserializeFromXml(string xml)
        {
            if (string.IsNullOrWhiteSpace(xml))
            {
                throw new ArgumentException("Input XML cannot be null or empty.");
            }

            var serializer = new System.Xml.Serialization.XmlSerializer(typeof(List<T>));
            using (var memoryStream = new MemoryStream(Encoding.UTF8.GetBytes(xml)))
            {
                var items = (List<T>)serializer.Deserialize(memoryStream) ?? new List<T>();
                var list = new ObservableList<T>();
                foreach (var item in items)
                {
                    list.Add(item);
                }
                return list;
            }
        }
        
        public IEnumerable<T> Filter(Func<T, bool> predicate)
        {
            if (predicate == null)
            {
                throw new ArgumentNullException(nameof(predicate), "Predicate cannot be null.");
            }

            int left = 0;
            int right = items.Count - 1;

            while (left <= right)
            {
                if (left == right)
                {
                    if (predicate(items[left]))
                    {
                        yield return items[left];
                    }
                    break;
                }

                if (predicate(items[left]))
                {
                    yield return items[left];
                }
                if (predicate(items[right]))
                {
                    yield return items[right];
                }

                left++;
                right--;
            }
        }

        
        public List<T> OrderBy(IComparer<T> comparer)
        {
            if (comparer == null)
            {
                throw new ArgumentNullException(nameof(comparer), "Comparer cannot be null.");
            }

            if (items.Count == 0)
            {
                return new List<T>(); // Vrátí prázdný seznam, pokud kolekce neobsahuje žádné položky
            }

            var sortedList = new List<T>(items);
            sortedList.Sort(comparer);
            return sortedList;
        }

        // Počet položek
        public int Count => items.Count;
        public IEnumerator<T> GetEnumerator()
        {
            int left = 0;
            int right = items.Count - 1;

            while (left <= right)
            {
                if (left == right)
                {
                    yield return items[left];
                    break;
                }

                yield return items[left];
                yield return items[right];
                left++;
                right--;
            }
        }
        // Iterace přes seznam
     
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();


    }
}
