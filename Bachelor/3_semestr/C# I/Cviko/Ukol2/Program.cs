namespace Ukol2
{
    public class Program
    {
        public enum ParseIntOption
        {
            NONE,
            ALLOW_WHITESPACES,
            ALLOW_NEGATIVE_NUMBERS,
            IGNORE_INVALID_CHARACTERS
        }
        static void Main()
        {
            
            string input = Console.ReadLine();
            Console.WriteLine(input + ' ' + input.GetType());

            int Intresult = ParseInt(input);
            Console.WriteLine(Intresult.ToString() + ' ' + Intresult.GetType());


            if (TryParseInt(input, out int result))
            {
                Console.WriteLine(result.ToString() + ' ' + Intresult.GetType());
            }
            else
            {
                Console.WriteLine("Neplatný vstup.");
            }

            ParseIntOption option = ParseIntOption.NONE;
            if (TryParseInt2(input, option, out result))
            {
                Console.WriteLine(result.ToString() + ' ' + Intresult.GetType());
            }
            else
            {
                Console.WriteLine("Neplatný vstup.");
            }

        }

        public static int ParseInt (string input)
        {
            if (string.IsNullOrEmpty(input))
                return -1;

            int result = 0;
            int sign = 1;
            int startIndex = 0;

            if (input[0] == '-')
            {
                sign = -1;
                startIndex = 1;
            }
            else if (input[0] == '+')
            {
                startIndex = 1;
            }

            for (int i = startIndex; i < input.Length; i++)
            {
                char c = input[i];

                if (c < '0' || c > '9')
                {
                    return -1;
                }

                result = result * 10 + (c - '0');
            }

            return result * sign;
        }

        public static int? ParseIntOrNull(string input) 
        {
            if (string.IsNullOrEmpty(input))
                return null;

            int result = 0;
            int sign = 1;
            int startIndex = 0;

            if (input[0] == '-')
            {
                sign = -1;
                startIndex = 1;
            }
            else if (input[0] == '+')
            {
                startIndex = 1;
            }

            for (int i = startIndex; i < input.Length; i++)
            {
                char c = input[i];

                if (c < '0' || c > '9')
                {
                    return null;
                }

                result = result * 10 + (c - '0');
            }

            return result * sign;
        }

        public static bool TryParseInt(string input, out int output)
        {
            output = 0;

            if (string.IsNullOrEmpty(input))
                return false;

            int result = 0;
            int sign = 1;
            int startIndex = 0;

            if (input[0] == '-')
            {
                sign = -1;
                startIndex = 1;
            }
            else if (input[0] == '+')
            {
                startIndex = 1;
            }

            for (int i = startIndex; i < input.Length; i++)
            {
                char c = input[i];

                if (c < '0' || c > '9')
                {
                    return false;
                }

                result = result * 10 + (c - '0');
            }

            output = result * sign;
            return true;
        }

        public static bool TryParseInt2(string input, ParseIntOption options, out int output)
        {
            output = 0;

            if (string.IsNullOrEmpty(input))
                return false;

            int result = 0;
            int sign = 1;
            int startIndex = 0;

            // Zpracování znaménka
            if (input[0] == '-')
            {
                if (options.HasFlag(ParseIntOption.ALLOW_NEGATIVE_NUMBERS))
                {
                    sign = -1;
                    startIndex = 1;
                }
                else
                {
                    return false; // Negativní čísla nejsou povolena
                }
            }
            else if (input[0] == '+')
            {
                startIndex = 1;
            }

            for (int i = startIndex; i < input.Length; i++)
            {
                char c = input[i];

                // Ignorovat bílé znaky, pokud je to povoleno
                if (char.IsWhiteSpace(c) && options.HasFlag(ParseIntOption.ALLOW_WHITESPACES))
                {
                    continue;
                }

                // Zpracování číslic
                if (char.IsDigit(c))
                {
                    result = result * 10 + (c - '0');
                }
                // Zpracování neplatných znaků
                else if (options.HasFlag(ParseIntOption.IGNORE_INVALID_CHARACTERS))
                {
                    // Pokud je neplatný znak a povoleno ignorování, pokračuj
                    continue;
                }
                else
                {
                    // Pokud není povoleno ignorování, vrať false
                    return false;
                }
            }

            output = result * sign;
            return true;
        }

    }
}
