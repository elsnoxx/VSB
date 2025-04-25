using apicalls;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace testDesktopApp
{
    public class CompanyItem
    {
        public string ico { get; set; }
        public string obchodniJmeno { get; set; }
        public string dic { get; set; }
        public string nazevObce { get; set; }
        public string poznamka { get; set; }

        public DateTime Added { get; set; } = DateTime.Now;
    }
}
