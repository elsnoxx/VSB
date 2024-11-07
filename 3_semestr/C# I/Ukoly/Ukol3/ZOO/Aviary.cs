using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public class Aviary : AnimalEnclosure
    {
        private List<IFlyable> animals = new List<IFlyable>();

        public Aviary(string name)
        {
            Name = name;
        }

        // Přidání zvířat, která umí létat
        public override void Add(Animal animal)
        {
            if (animal is IFlyable flyableAnimal)
            {
                animals.Add(flyableAnimal);
            }
            else
            {
                Console.WriteLine("Toto zvíře nemůže být přidáno do voliéry (nemá schopnost létat).");
            }
        }

        // Vydávání zvuků všech zvířat v voliéře
        public override void MakeAnimalsSound()
        {
            Console.WriteLine($"Voliéra {Name}: zvuky zvířat");
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
