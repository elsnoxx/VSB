using Ukol7;

public class Program
{
    public static void Main()
    {
        int[] numbers = { 1, 2, 3, 4 };
        ArrayHelper.Swap<int>(numbers, 0, 3);
        Console.WriteLine("Po záměně: " + string.Join(", ", numbers));

        string[] words1 = { "Hello", "world" };
        string[] words2 = { "from", "C#" };
        string[] combinedWords = ArrayHelper.Concat<string>(words1, words2);
        Console.WriteLine("Po spojení: " + string.Join(" ", combinedWords));
    }
}