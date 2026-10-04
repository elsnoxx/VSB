using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public class Pond : AnimalEnclosure
    {

        public Pond(string name)
        {
            Name = name;
        }

        // Přidání zvířat, která umí plavat
        public override void Add(Animal animal)
        {
            if (animal is ISwimmable swimmableAnimal)
            {
                animals.Add(animal);
            }
            else
            {
                Console.WriteLine("Toto zvíře nemůže být přidáno do rybníka (nemá schopnost plavat).");
            }
        }

        // Vydávání zvuků všech zvířat v rybníku
        public override void MakeAnimalsSound()
        {
            Console.WriteLine($"Rybník {Name}:");
            foreach (var animal in animals)
            {
                if (animal is ISoundEmitter soundEmitter)
                {
                    soundEmitter.MakeSound();
                }
            }
        }

    }
}
