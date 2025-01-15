using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace RealTimeTest_1
{
    public class Peak
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }
        public double Lon { get; set; }


        public double Lat { get; set; }

        [JsonPropertyName("ele")]
        public double? Elevation { get; set; }

        public override string ToString()
        {
            if (Elevation != null)
            {
                return ($"VRCHOL | {Name} ({Lat}, {Lon}): {Elevation}").ToString();
            }
            else
            {
                return ($"VRCHOL | {Name} ({Lat}, {Lon}): ???m.n.m").ToString();
            }
            
        }

    }
}
