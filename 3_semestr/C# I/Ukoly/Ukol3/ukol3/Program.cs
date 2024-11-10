using ZOO.Animals;
using ZOO;
namespace ukol3;

class Program
{
    static void Main(string[] args)
    {
        List<IAnimalEnclosure> animalEnclosures = new List<IAnimalEnclosure>();

        var pond = new Pond("Velký rybník");
        var mixedAviary = new Aviary("Smíšená voliéra");
        var owlAviary = new Aviary("Sovy");


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
            List<IFlyable> flyables = enclosure.GetAnimals<Animal>().OfType<IFlyable>().ToList();
            foreach (IFlyable animal in flyables)
            {
                animal.Fly();
            }

            Console.WriteLine();
            Console.WriteLine($"{enclosure.Name} - plavání:");
            Console.WriteLine("-------------");
            List<ISwimmable> swimmables = enclosure.GetAnimals<Animal>().OfType<ISwimmable>().ToList();
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