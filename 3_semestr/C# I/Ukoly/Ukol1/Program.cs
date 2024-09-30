using System.Globalization;

namespace Ukol1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = CultureInfo.GetCultureInfo("cs-CZ");
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            //string data = File.ReadAllText(args[0]);
            string data = File.ReadAllText(@"C:\\Users\\ficek\\source\repos\\Ukol1\\data.txt");

            //Product[] products = 

            ParseData(data);
        }

        public static void ParseData(string data)
        {
            data = data.Trim(' ');
            int index = 0;
            string[] lines = data.Split('\n');
            foreach (string line in lines) 
            {
                string test = line.Trim();
                if (!string.IsNullOrEmpty(test))
                {
                    // Console.WriteLine(test);
                    if (line.StartsWith("- price:"))
                    {
                        index = line.IndexOf("price:");
                        if (index != -1)
                        {
                            Console.WriteLine("price " + line.Substring(index + "price:".Length).Trim());
                        }
                    }
                    else if (line.StartsWith("- quantity:"))
                    {
                        index = line.IndexOf("quantity:");
                        if (index != -1)
                        {
                            Console.WriteLine("quantity " + line.Substring(index + "quantity:".Length).Trim());
                        }
                    }
                    else if (line.StartsWith("- weight:"))
                    {
                        index = line.IndexOf("weight:");
                        if (index != -1)
                        {
                            Console.WriteLine("weight " + line.Substring(index + "weight:".Length).Trim());
                        }
                    }
                }
                
            }
            

        }
    }
}
