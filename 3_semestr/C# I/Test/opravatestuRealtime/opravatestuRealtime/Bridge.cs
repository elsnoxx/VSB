using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opravatestuRealtime
{
    public class Bridge: Attraction
    {
        public double Length { get; set; }


        public override string ToString()
        {
            return $"Most - {Name} | delka {Length}";
        }
    }
}
