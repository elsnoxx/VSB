using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace RealTimeTest_1
{
    
    public class Uzel
    {
        [JsonPropertyName("natural")]
        public string natural { get; set; }
        [JsonPropertyName("shop")]
        public string shop { get; set; }

        public Uzel()
        {

        }
        
        
        //public <T> ToPlace()
        //{
        //    if (natural == null)
        //    {
        //        return new Shop();
        //    }
        //    else if ( shop == null)
        //    {
        //        return new Peak();
        //    }
        //}
    }
}
