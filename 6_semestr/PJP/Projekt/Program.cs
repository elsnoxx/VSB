using Antlr4.Runtime;
using Antlr4.Runtime.Tree;
using System.Globalization;
using System.Runtime.CompilerServices;

namespace Projekt
{
    internal class Program
    {
        static readonly string[] fileNames = ["PLC_t1.in.txt", "PLC_t2.in.txt", "PLC_t3.in.txt", "PLC_errors.in.txt"];
        static void Main(string[] args)
        {
            Thread.CurrentThread.CurrentCulture = new CultureInfo("en-US");
            //var fileName = "input.txt";
            //Console.WriteLine("Parsing: " + fileName);

            //if (args.Length > 0)
            //{
            //    fileName = args[0];
            //    Console.WriteLine("Parsing: " + fileName);
            //    CompilationProces(fileName);
            //} else {
            //    Console.WriteLine("No input file provided. Processing default test files...");
            //}

            foreach (var file in fileNames)
            {
                Console.WriteLine($"\n--- Processing {file} ---");
                CompilationProces(file);
            }

        }

        public static void CompilationProces(string fileName)
        {
            var input = File.ReadAllText(fileName);
            var inputStream = new AntlrInputStream(input);
            var lexer = new PLCProjectLexer(inputStream);
            var tokenStream = new CommonTokenStream(lexer);
            var parser = new PLCProjectParser(tokenStream);

            // 1. Syntaktická analýza
            var errorListener = new VerboseListener();
            parser.RemoveErrorListeners();
            parser.AddErrorListener(errorListener);

            var tree = parser.prog();

            if (errorListener.HadError)
            {
                Console.WriteLine("Computation stopped due to syntax errors.");
                return;
            }

            // 2. Kontrola typů
            var typeChecker = new TypeCheckingVisitor();
            typeChecker.Visit(tree);

            if (typeChecker.Errors.Any())
            {
                foreach (var err in typeChecker.Errors) Console.WriteLine(err);
                Console.WriteLine("Computation stopped due to type errors.");
                return;
            }

            // 3. Generování kódu
            Console.WriteLine("Type checking passed! Generating code...");

            var codeGen = new CodeGeneratorVisitor(typeChecker);
            string finalCode = codeGen.Visit(tree);

            // Vyčištění prázdných řádků a výpis
            finalCode = string.Join("\n", finalCode.Split('\n', StringSplitOptions.RemoveEmptyEntries));
            Console.WriteLine("--- GENERATED CODE ---");
            //Console.WriteLine(finalCode);

            // Volitelně uložení do souboru
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
