using System.Diagnostics;

namespace opravatestuRealtime
{
    internal class Program
    {
        static void Main(string[] args)
        {
            AttractionList list = new AttractionList();

            list.Load("attractions.bin");

            foreach (var item in list)
            {
                Console.WriteLine(item);
            }

            list.Save("attractions-output.txt");
        }
    }
}
