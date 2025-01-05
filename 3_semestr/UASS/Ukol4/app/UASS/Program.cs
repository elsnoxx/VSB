using System;
using Network;

namespace UASS
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Graph graph = new Graph();

            // Načtení grafu ze souboru
            string filePath = @"C:\Users\ficek\Documents\GitHub\VSB\3_semestr\UASS\Ukol4\app\email-dnc\email-dnc.edges";
            graph.LoadFromFile(filePath);

            // Export kumulativních snímků
            string outputDirectory = @"C:\Users\ficek\Documents\GitHub\VSB\3_semestr\UASS\Ukol4\GraphSnapshots";
            int numberOfSnapshots = 5; // Počet kumulativních snímků

            graph.ExportCumulativeSnapshots(outputDirectory, numberOfSnapshots, "yyyyMM");

            Console.WriteLine("Cumulative snapshots have been created successfully.");
        }
    }
}
