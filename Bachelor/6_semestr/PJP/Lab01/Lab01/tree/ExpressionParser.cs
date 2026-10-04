using Lab01.tree;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace Lab01.tree
{
    public class ExpressionParser
    {
        static char[] symbols = { '+', '-', '*', '/' };

        public static Node Parse(string text)
        {
            // remove whitespace
            text = Regex.Replace(text, @"\s+", "");

            if (text.Length == 0)
            {
                Console.WriteLine("Empty expression");
                return null;
            }
                
            

            ValidateExpression(text);
            
            if (text.StartsWith("(") &&
                text.EndsWith(")") &&
                IsValidBrackets(text.Substring(1, text.Length - 2)))
            {
                return Parse(text.Substring(1, text.Length - 2));
            }

            int lastOperatorIndex = -1;
            int minPriority = 3;
            int bracketLevel = 0;

            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];

                if (c == '(') bracketLevel++;
                else if (c == ')') bracketLevel--;

                if (bracketLevel == 0 && symbols.Contains(c))
                {
                    int priority = GetOperatorPriority(c);

                    if (priority <= minPriority)
                    {
                        minPriority = priority;
                        lastOperatorIndex = i;
                    }
                }
            }

            if (lastOperatorIndex != -1)
            {
                return new BinaryOperator(
                    text[lastOperatorIndex],
                    Parse(text.Substring(0, lastOperatorIndex)),
                    Parse(text.Substring(lastOperatorIndex + 1))
                );
            }

            if (double.TryParse(text, out double value))
            {
                return new Value(value);
            }

            Console.WriteLine("Invalid expression");
            return null;
        }
        
        static void ValidateExpression(string text)
        {
            for (int i = 0; i < text.Length - 1; i++)
            {
                if (IsOperator(text[i]) && IsOperator(text[i + 1]))
                {
                    Console.WriteLine($"Neplatná sekvence operátorů: {text[i]}{text[i + 1]}");
                }
            }
        }

        static bool IsOperator(char c)
        {
            return c == '+' || c == '-' || c == '*' || c == '/';
        }
        public static double Evaluate(Node node)
        {
            if (node is Value v)
            {
                return v.ValueNumber;
            }

            if (node is BinaryOperator op)
            {
                switch (op.Operator)
                {
                    case '+':
                        return Evaluate(op.Left) + Evaluate(op.Right);
                    case '-':
                        return Evaluate(op.Left) - Evaluate(op.Right);
                    case '*':
                        return Evaluate(op.Left) * Evaluate(op.Right);
                    case '/':
                        return Evaluate(op.Left) / Evaluate(op.Right);
                }
            }

            Console.WriteLine("Invalid node");
            return double.NaN;
        }
        static bool IsValidBrackets(string text)
        {
            int count = 0;

            foreach (char c in text)
            {
                if (c == '(') count++;
                if (c == ')') count--;

                if (count < 0)
                    return false;
            }

            return count == 0;
        }

        static int GetOperatorPriority(char op)
        {
            return (op == '+' || op == '-') ? 1 : 2;
        }
    }
}
