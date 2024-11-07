using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public class Pond : AnimalEnclosure
    {
        private List<ISwimmable> animals = new List<ISwimmable>();

        public Pond(string name)
        {
            Name = name;
        }

        // Přidání zvířat, která umí plavat
        public override void Add(Animal animal)
        {
            if (animal is ISwimmable swimmableAnimal)
            {
                animals.Add(swimmableAnimal);
            }
            else
            {
                Console.WriteLine("Toto zvíře nemůže být přidáno do rybníka (nemá schopnost plavat).");
            }
        }

        // Vydávání zvuků všech zvířat v rybníku
        public override void MakeAnimalsSound()
        {
            Console.WriteLine($"Rybník {Name}: zvuky zvířat");
            foreach (var animal in animals)
            {
                if (animal is ISoundEmitter soundEmitter)
                {
                    soundEmitter.MakeSound();
                }
            }
        }

        // Získání zvířat požadovaného typu
        public override List<T> GetAnimals<T>()
        {
            return animals.OfType<T>().ToList();
        }
    }
}
