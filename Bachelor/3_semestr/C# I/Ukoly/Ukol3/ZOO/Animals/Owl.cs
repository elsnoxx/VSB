using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZOO.Animals
{
    public class Owl : Animal, IFlyable, IBird
    {
        public Owl(string name) : base(name) { }

        public void Fly()
        {
            Console.WriteLine($"Sova {Name} letí.");
        }

        public void MakeSound()
        {
            Console.WriteLine($"Sova {Name}: húúú, húúú");
        }
    }
}
