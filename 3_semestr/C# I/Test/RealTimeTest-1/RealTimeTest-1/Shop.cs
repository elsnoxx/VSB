using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace RealTimeTest_1
{
    public class Shop
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }
        public double Lon { get; set; }

        public double Lat { get; set; }

        [JsonPropertyName("opening_hours")]
        public string OpeningHours { get; set; }

        [JsonPropertyName("shop")]
        public string Type { get; set; }

        public Shop() { }

        public override string ToString()
        {
            if (OpeningHours != null)
            {
                return ($"OBCHOD | {Name} ({Lat}, {Lon}): {Type} - {OpeningHours}").ToString();
            }
            else
            {
                return ($"OBCHOD | {Name} ({Lat}, {Lon}): {Type} - ???").ToString();
            }

        }

    }
}
