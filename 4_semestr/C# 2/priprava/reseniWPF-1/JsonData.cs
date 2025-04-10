using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace reseniWPF_1
{
    public class JsonData
    {
        public string obchodniJmeno { get; set; }
        public string dic { get; set; }
        public Sidlo sidlo { get; set; }

        public override string ToString()
        {
            return $"{obchodniJmeno}, {sidlo.nazevObce}, {dic}";
        }
    }

    public class Sidlo
    {
        public string nazevObce { get; set; }
    }
}
