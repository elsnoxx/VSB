using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ZOO.Animals
{
    public class Duck : Animal, IBird, ISwimmable, IFlyable
    {
        public Duck(string name) : base(name) { }

        public void Fly()
        {
            Console.WriteLine($"Kachna {Name} letí.");
        }

        public void Swim()
        {
            Console.WriteLine($"Kachna {Name} plave.");
        }
        public void MakeSound()
        {
            Console.WriteLine($"Kachna {Name}: kvak, kvak");
        }
    }
}
