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
            int n = int.Parse(Console.ReadLine());

            for (int i = 0; i < n; i++)
            {
                string expression = Console.ReadLine();
                var node = ExpressionParser.Parse(expression);
                double result = ExpressionParser.Evaluate(node);
                Console.WriteLine(result);
            }
        }
    }
}
