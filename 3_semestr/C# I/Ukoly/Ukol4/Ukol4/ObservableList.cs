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
        public List<T> OrderBy(IComparer<T> comparer)
        {
            var sortedList = new List<T>(items);
            sortedList.Sort(comparer);
            return sortedList;
        }

        public string SerializeToJson()
        {
            JsonSerializerOptions options = new JsonSerializerOptions();
            options.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
            options.Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;

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

        public void DeserializeFromJson(string json)
        {
            if (string.IsNullOrWhiteSpace(json))
            {
                throw new ArgumentException("Input JSON cannot be null or empty.");
            }

            items = JsonSerializer.Deserialize<List<T>>(json) ?? new List<T>();
        }


        public void DeserializeFromXml(string xml)
        {
            if (string.IsNullOrWhiteSpace(xml))
            {
                throw new ArgumentException("Input XML cannot be null or empty.");
            }

            var serializer = new System.Xml.Serialization.XmlSerializer(typeof(List<T>));

            using (var memoryStream = new MemoryStream(Encoding.UTF8.GetBytes(xml)))
            {
                items = (List<T>)serializer.Deserialize(memoryStream) ?? new List<T>();
            }
        }

        // Počet položek
        public int Count => items.Count;

        // Iterace přes seznam
        public IEnumerator<T> GetEnumerator() => items.GetEnumerator();
        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();


    }
}
