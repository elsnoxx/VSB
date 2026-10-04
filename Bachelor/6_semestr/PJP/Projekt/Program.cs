using Antlr4.Runtime;
using Antlr4.Runtime.Tree;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Projekt
{
    internal class Program
    {
        //static readonly string[] fileNames = ["PLC_t1.in.txt", "PLC_t2.in.txt", "PLC_t3.in.txt", "PLC_errors.in.txt"];
        static readonly string[] fileNames = ["PLC_t4.in.txt"];
        static string fileName = "";
        static bool isVerbose = true;
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            //var fileName = "input.txt";
            //Console.WriteLine("Parsing: " + fileName);

            //if (args.Length == 0)
            //{
            //    PrintHelp();
            //    return;
            //}

            // Kontrola, zda je přítomen argument -v
            //isVerbose = args.Contains("-v");

            // Získání cesty k souboru (vezmeme první argument, který není -v)
            //string fileName = args.FirstOrDefault(a => a != "-v");

            //if (string.IsNullOrEmpty(fileName))
            //{
            //    Console.WriteLine("Error: No input file specified.");
            //    PrintHelp();
            //    return;
            //}

            //if (File.Exists(fileName))
            //{
            //    CompilationProces(fileName, isVerbose);
            //}
            //else
            //{
            //    Console.WriteLine($"Error: File '{fileName}' not found.");
            //}

            // automatické zpracování všech testovacích souborů
            foreach (var file in fileNames)
            {
                Console.WriteLine($"\n--- Processing {file} ---");
                CompilationProces(file, isVerbose);
            }

        }

        public static void PrintHelp()
        {
            Console.WriteLine("Usage: dotnet run <input_file> [-v]");
            Console.WriteLine("Options:");
            Console.WriteLine("  -v    Verbose mode (prints tokens, parse tree and generated code)");
        }

        public static void CompilationProces(string fileName, bool verbose)
        {
            var input = File.ReadAllText(fileName);
            var inputStream = new AntlrInputStream(input);
            var lexer = new PLCProjectLexer(inputStream);
            var tokenStream = new CommonTokenStream(lexer);

            // --- VERBOSE: VÝPIS TOKENŮ ---
            if (verbose)
            {
                Console.WriteLine("\n--- TOKENS ---");
                tokenStream.Fill();
                foreach (var token in tokenStream.GetTokens())
                {
                    Console.WriteLine(token.ToString());
                }
                tokenStream.Reset();
            }

            var parser = new PLCProjectParser(tokenStream);
            var errorListener = new VerboseListener();
            parser.RemoveErrorListeners();
            parser.AddErrorListener(errorListener);

            var tree = parser.prog();

            // --- VERBOSE: VÝPIS STROMU ---
            if (verbose)
            {
                Console.WriteLine("\n--- PARSE TREE ---");
                Console.WriteLine(tree.ToStringTree(parser));
            }

            if (errorListener.HadError)
            {
                Console.WriteLine("\nComputation stopped due to syntax errors.");
                return;
            }

            var typeChecker = new TypeCheckingVisitor();
            typeChecker.Visit(tree);

            if (typeChecker.Errors.Any())
            {
                foreach (var err in typeChecker.Errors) Console.WriteLine(err);
                Console.WriteLine("\nComputation stopped due to type errors.");
                return;
            }

            var codeGen = new CodeGeneratorVisitor(typeChecker);
            string finalCode = codeGen.Visit(tree);
            finalCode = string.Join("\n", finalCode.Split('\n', StringSplitOptions.RemoveEmptyEntries));

            // --- VERBOSE: VÝPIS GENEROVANÉHO KÓDU ---
            if (verbose)
            {
                Console.WriteLine("\n--- GENERATED CODE ---");
                Console.WriteLine(finalCode);
                Console.WriteLine("----------------------\n");
            }

            File.WriteAllText(fileName + ".out", finalCode);

            try
            {
                Console.WriteLine("--- STARTING VIRTUAL MACHINE ---");
                VirtualMachine vm = new VirtualMachine(finalCode);
                vm.Run();
                Console.WriteLine("--- VM FINISHED SUCCESSFULLY ---");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"VM Runtime Error: {ex.Message}");
            }
        }
    }
}
