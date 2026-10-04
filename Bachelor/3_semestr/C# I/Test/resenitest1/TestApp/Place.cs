using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestApp
{
    internal class Place
    {
        public double Lat { get; set; }
        public double Lon { get; set; }

        private string name;
        public string Name {
            get { return name ?? "Neznámý název"; }
            set { name = value; }
        }
    }
}
