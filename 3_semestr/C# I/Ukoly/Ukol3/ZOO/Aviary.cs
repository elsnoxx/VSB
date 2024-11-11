using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public class Aviary : AnimalEnclosure
    {

        public Aviary(string name)
        {
            Name = name;
        }

        // Přidání zvířat, která umí létat
        public override void Add(Animal animal)
        {
            if (animal is IFlyable flyableAnimal)
            {
                animals.Add(animal);
            }
            else
            {
                Console.WriteLine("Toto zvíře nemůže být přidáno do voliéry (nemá schopnost létat).");
            }
        }

        // Vydávání zvuků všech zvířat v voliéře
        public override void MakeAnimalsSound()
        {
            Console.WriteLine($"Voliéra {Name}:");
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
