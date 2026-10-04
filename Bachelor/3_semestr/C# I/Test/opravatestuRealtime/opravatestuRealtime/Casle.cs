using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opravatestuRealtime
{
    public class Casle : Attraction, IOpeningHours
    {
        public int? OpenTime { get; set; }
        public int? CloseTime { get; set; }



        public override string ToString()
        {
            if (OpenTime == null && CloseTime == null)
            {
                return $"Hrad - {Name} | oteviraci doba ? - ?";
            }

            if (OpenTime == null)
            {
                return $"Hrad - {Name} | oteviraci doba ? - {CloseTime}";
            }

            if (CloseTime == null)
            {
                return $"Hrad - {Name} | oteviraci doba {OpenTime} - ?";
            }

            return $"Hrad - {Name} | oteviraci doba {OpenTime} - {CloseTime}";
        }
    }

}
