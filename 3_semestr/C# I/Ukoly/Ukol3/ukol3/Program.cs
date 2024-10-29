namespace ukol3;

class Program
{
    static void Main(string[] args)
    {
        List<IAnimalEnclosure> animalEnclosures = new List<IAnimalEnclosure>();

        var pond = /* Vytvoření instance rybníku s názvem "Velký rybník" */;
        var mixedAviary = /* Vytvoření instance voliéry s názvem "Smíšená voliéra" */;
        var owlAviary = /* Vytvoření instance voliéry s názvem "Sovy" */;


        animalEnclosures.Add(pond);
        animalEnclosures.Add(mixedAviary);
        animalEnclosures.Add(owlAviary);

        pond.Add(new Duck("Kvaky"));
        pond.Add(new Catfish("Pepa"));
        mixedAviary.Add(new Owl("Šedivka"));
        mixedAviary.Add(new Duck("Donald"));

        owlAviary.Add(new Owl("Houk"));
        owlAviary.Add(new Owl("Sovík"));



        // ----

        foreach (IAnimalEnclosure enclosure in animalEnclosures)
        {
            Console.WriteLine($"{enclosure.Name} - let:");
            Console.WriteLine("-------------");
            List<IFlyable> flyables = /* získání seznamu všech IFlyable pomocí metody GetAnimals na výběhu */;
            foreach (IFlyable animal in flyables)
            {
                animal.Fly();
            }

            Console.WriteLine();
            Console.WriteLine($"{enclosure.Name} - plavání:");
            Console.WriteLine("-------------");
            List<ISwimmable> swimmables = /* získání seznamu všech ISwimmable pomocí metody GetAnimals na výběhu */;
            foreach (ISwimmable animal in swimmables)
            {
                animal.Swim();
            }

            Console.WriteLine();
            Console.WriteLine();
            Console.WriteLine();
        }



        Console.WriteLine();
        Console.WriteLine();
        Console.WriteLine("Zvuky zvířat:");
        Console.WriteLine("-------------");
        foreach (var enclosure in animalEnclosures)
        {
            enclosure.MakeAnimalsSound();
        }
    }
}