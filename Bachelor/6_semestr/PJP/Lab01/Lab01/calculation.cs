using System.Collections.Generic;

namespace Lab01
{
    public class calculation
    {
        public static double Evaluate(string expr)
        {
            expr = expr.Replace(" ", "");
            

            while (expr.Contains("("))
            {
                int start = expr.LastIndexOf('(');
                int end = expr.IndexOf(')', start);

                string inside = expr.Substring(start + 1, end - start - 1);

                double result = EvaluateSimple(inside);

                expr = expr.Substring(0, start) + result + expr.Substring(end + 1);
            }

            return EvaluateSimple(expr);
        }
        
        static double EvaluateSimple(string expr)
        {
            expr = expr.Replace(" ", "");

            List<string> tokens = new List<string>();
            string number = "";

            foreach (char c in expr)
            {
                if (char.IsDigit(c))
                    number += c;
                else
                {
                    tokens.Add(number);
                    tokens.Add(c.ToString());
                    number = "";
                }
            }

            tokens.Add(number);

            // first * /
            for (int i = 0; i < tokens.Count; i++)
            {
                if (tokens[i] == "*" || tokens[i] == "/")
                {
                    double left = double.Parse(tokens[i - 1]);
                    double right = double.Parse(tokens[i + 1]);

                    double result = tokens[i] == "*" ? left * right : left / right;

                    tokens[i - 1] = result.ToString();
                    tokens.RemoveAt(i);
                    tokens.RemoveAt(i);
                    i--;
                }
            }

            // second + -
            double value = double.Parse(tokens[0]);

            for (int i = 1; i < tokens.Count; i += 2)
            {
                double next = double.Parse(tokens[i + 1]);

                if (tokens[i] == "+")
                    value += next;
                else
                    value -= next;
            }

            return value;
        }
    }
}