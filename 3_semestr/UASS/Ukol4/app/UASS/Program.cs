using System;
using Network;

namespace UASS
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Graph graph = new Graph();

            string filePath = @"C:\Users\ficek\OneDrive\Dokumenty\GitHub\VSB\3_semestr\UASS\Ukol4\app\email-dnc\email-dnc.edges";
            graph.LoadFromFile(filePath);

            string outputDirectory = @"C:\Users\ficek\OneDrive\Dokumenty\GitHub\VSB\3_semestr\UASS\Ukol4\snapshots";
            graph.ExportSnapshots(outputDirectory, "yyyyMM");

            Console.WriteLine("Snapshots have been created successfully.");
        }
    }
}
