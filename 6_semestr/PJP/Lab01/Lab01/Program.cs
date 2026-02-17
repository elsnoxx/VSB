using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab01
{
    internal class Program
    {
        
        static void Main(string[] args)
        {
            string input = Console.ReadLine();

            int open = 0;
            int close = 0;

            input = input.Trim();
            List<char> tokens = new List<char>();

            for (int i = 0; i < input.Length; i++)
            {
                if (!char.IsWhiteSpace(input[i]) )
                {

                    tokens.Add(input[i]);
                }
                else
                {
                    Console.WriteLine("ERROR");
                }
            }
            foreach (var item in tokens)
            {
                Console.WriteLine(item);
            }
        }
    }
}
