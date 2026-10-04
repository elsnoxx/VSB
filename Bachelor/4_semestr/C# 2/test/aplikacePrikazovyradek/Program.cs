using apicalls;

namespace aplikacePrikazovyradek
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            string ico = GetIcoFromArgsOrInput(args);

            try
            {
                var company = await ARESData.GetCompanyDataAsync(ico.Trim());

                if (company == null)
                {
                    Console.WriteLine($"Firma s IČO {ico} nebyla nalezena.");
                }
                else
                {
                    PrintCompanyInfo(company);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Došlo k chybě: {ex.Message}");
            }
        }

        private static string GetIcoFromArgsOrInput(string[] args)
        {
            if (args.Length > 0 && !string.IsNullOrWhiteSpace(args[0]))
            {
                return args[0];
            }

            Console.WriteLine("Zadejte IČO firmy:");
            string? input = Console.ReadLine();

            while (string.IsNullOrWhiteSpace(input))
            {
                Console.WriteLine("IČO nesmí být prázdné. Zadejte prosím platné IČO:");
                input = Console.ReadLine();
            }

            return input;
        }

        private static void PrintCompanyInfo(Company company)
        {
            Console.WriteLine("====================================");
            Console.WriteLine("Informace o firmě:");
            Console.WriteLine($"Název: {company.obchodniJmeno}");
            Console.WriteLine($"Obec: {company.sidlo.nazevObce}");
            Console.WriteLine($"IČO: {company.ico}");
            Console.WriteLine($"DIČ: {company.dic}");
            Console.WriteLine("====================================");
        }
    }
}