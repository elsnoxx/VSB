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

        public void ExportCumulativeSnapshots(string outputDirectory, int numberOfSnapshots, string format = "yyyyMM")
        {
            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            var sortedNodes = Nodes.OrderBy(node => node.Timestamp).ToList();

            int totalNodes = sortedNodes.Count;
            int nodesPerSnapshot = totalNodes / numberOfSnapshots;

            List<Point> cumulativeNodes = new List<Point>();

            for (int i = 1; i <= numberOfSnapshots; i++)
            {
                int currentSnapshotLimit = Math.Min(i * nodesPerSnapshot, totalNodes);
                cumulativeNodes.AddRange(sortedNodes.Take(currentSnapshotLimit));

                string filePath = Path.Combine(outputDirectory, $"cumulative_snapshot_{i}.csv");

                using (StreamWriter writer = new StreamWriter(filePath))
                {
                    writer.WriteLine("Source,Target,Weight");

                    foreach (var node in cumulativeNodes)
                    {
                        writer.WriteLine($"{node.X},{node.Y},1");
                    }
                }

                Console.WriteLine($"Cumulative snapshot {i} exported to {filePath}");
            }
        }
    }
}
