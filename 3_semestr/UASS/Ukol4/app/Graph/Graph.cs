using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace Network
{
    public class Graph
    {
        public List<Point> Nodes { get; private set; }

        public Graph()
        {
            Nodes = new List<Point>();
        }

        public void AddPoint(Point point)
        {
            Nodes.Add(point);
        }

        public void LoadFromFile(string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine("File not found.");
                return;
            }

            using (FileStream fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (StreamReader streamReader = new StreamReader(fileStream))
            {
                string data = streamReader.ReadToEnd();
                string[] lines = data.Split(new[] { "\r\n", "\r", "\n" }, StringSplitOptions.None);

                foreach (var line in lines)
                {
                    if (!string.IsNullOrWhiteSpace(line))
                    {
                        string[] splitedLine = line.Split(',');
                        if (splitedLine.Length == 3)
                        {
                            if (int.TryParse(splitedLine[0], out int x) &&
                                int.TryParse(splitedLine[1], out int y) &&
                                long.TryParse(splitedLine[2], out long timestamp))
                            {
                                AddPoint(new Point(x, y, timestamp));
                            }
                            else
                            {
                                Console.WriteLine($"Invalid line format: {line}");
                            }
                        }
                        else
                        {
                            Console.WriteLine("Line error: Incorrect number of elements.");
                        }
                    }
                }
            }
        }

        public List<string> GetUniqueTimestamps(string format = "yyyyMM")
        {
            HashSet<string> uniqueTimestamps = new HashSet<string>();

            foreach (var node in Nodes)
            {
                uniqueTimestamps.Add(node.Timestamp.ToString(format));
            }

            List<string> sortedTimestamps = new List<string>(uniqueTimestamps);
            sortedTimestamps.Sort();

            return sortedTimestamps;
        }

        public void ExportSnapshots(string outputDirectory, string format = "yyyyMM")
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            // Rozdělení grafu na snímky podle časového formátu
            var groupedByTimestamps = Nodes.GroupBy(node => node.Timestamp.ToString(format));

            foreach (var group in groupedByTimestamps)
            {
                string timestamp = group.Key;
                string filePath = Path.Combine(outputDirectory, $"graph_snapshot_{timestamp}.csv");

                using (StreamWriter writer = new StreamWriter(filePath))
                {
                    writer.WriteLine("Source,Target,Weight"); // Hlavička CSV

                    foreach (var node in group)
                    {
                        writer.WriteLine($"{node.X},{node.Y},1"); // Weight je zde 1 jako příklad
                    }
                }

                Console.WriteLine($"Snapshot for {timestamp} exported to {filePath}");
            }
        }
    }
}
