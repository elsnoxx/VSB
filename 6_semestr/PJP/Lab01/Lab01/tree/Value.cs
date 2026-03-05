using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lab01.tree
{
    public class Value : Node
    {
        public double ValueNumber { get; set; }

        public Value(double value)
        {
            ValueNumber = value;
        }
    }
}
