using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Lab01.tree;

namespace Lab01
{
    internal class Program
    {
        
        static void Main(string[] args)
        {
            /*
            int n = int.Parse(Console.ReadLine());

            for (int i = 0; i < n; i++)
            {
                string expression = Console.ReadLine();
                
                HandleCalculation(expression);
            }p
            */
            LoadFromFile("example.txt");
            
        }

        public static void LoadFromFile(string path)
        {
            if (!File.Exists(path))
            {
                Console.WriteLine("File not found");
                return;
            }

            string[] lines = File.ReadAllLines(path);
            
            int count = int.Parse(lines[0]);

            for (int i = 1; i <= count; i++)
            {
                string expression = lines[i];

                HandleCalculation(expression);
            }
        }

        public static void HandleCalculation(string expression)
        {
            if (ValidateExpression(expression) != 1)
            {
                try
                {
                    var node = ExpressionParser.Parse(expression);
                    double result = ExpressionParser.Evaluate(node);
                    Console.WriteLine($"Result from binary tree {expression}: {result}");

                    Console.WriteLine($"Evaluate by (): {calculation.Evaluate(expression)}");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Chyba: {ex.Message}");
                }
            }
            else
            {
                Console.WriteLine($"Expresion in bad format {expression}");
            }
        }
        
        static int ValidateExpression(string expr)
        {
            for (int i = 0; i < expr.Length - 1; i++)
            {
                if (IsOperator(expr[i]) && IsOperator(expr[i + 1]))
                {
                    // for debug print
                    //Console.WriteLine($"Neplatná sekvence operátorů: {expr[i]}{expr[i + 1]}");
                    return 1;
                }
            }

            return 0;
        }

        static bool IsOperator(char c)
        {
            return c == '+' || c == '-' || c == '*' || c == '/';
        }
    }
}
