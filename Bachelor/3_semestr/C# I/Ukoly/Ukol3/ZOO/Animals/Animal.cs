using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZOO.Animals
{
    public abstract class Animal
    {
        public string Name { get; init; }

        public Animal(string name) 
        {
            Name = name;
        }
    }
}
