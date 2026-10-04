using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace ZOO.Animals
{
    public class Catfish : Animal, ISwimmable
    {
        public Catfish(string name) : base(name) { }

        public void Swim()
        {
            Console.WriteLine($"Sumec {Name} plave.");
        }
    }
}
