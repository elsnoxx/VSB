using System;
using System.Collections.Generic;
using ZOO.Animals;

namespace ZOO
{
    public abstract class AnimalEnclosure : IAnimalEnclosure
    {
        public string Name { get; protected set; }
        protected List<Animal> animals = new List<Animal>();
        public abstract void Add(Animal animal);
        public abstract void MakeAnimalsSound();

        public virtual List<T> GetAnimals<T>() where T : class
        {
            return animals.OfType<T>().ToList();
        }
    }
}
