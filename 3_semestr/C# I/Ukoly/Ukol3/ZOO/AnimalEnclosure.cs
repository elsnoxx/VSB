using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public abstract class AnimalEnclosure : IAnimalEnclosure
    {
        public string Name { get; protected set; }
        public abstract void Add(Animal animal);
        public abstract void MakeAnimalsSound();

        public abstract List<T> GetAnimals<T>() where T : Animal;
    }
}
