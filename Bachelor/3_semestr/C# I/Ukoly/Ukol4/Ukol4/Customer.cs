using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol4
{
    public class Customer
    {
        public string Name { get; set; }
        public int Age {  get; set; }

        public Customer()
        {
        }

        public Customer(string name, int age) 
        { 
            this.Name = name;
            this.Age = age;
        }

        public override string ToString()
        {
            return $"{Name} ({Age})";
        }
    }
}
