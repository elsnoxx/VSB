using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opravatestuRealtime
{
    internal class Aquapark : Attraction, IOpeningHours
    {
        public int? OpenTime { get; set; }
        public int? CloseTime { get; set; }


        public override string ToString()
        {
            if (OpenTime == null && CloseTime == null)
            {
                return $"Aqupark - {Name} | oteviraci doba ? - ?";
            }

            if (OpenTime == null)
            {
                return $"Aqupark - {Name} | oteviraci doba ? - {CloseTime}";
            }

            if (CloseTime == null)
            {
                return $"Aqupark - {Name} | oteviraci doba {OpenTime} - ?";
            }

            return $"Aqupark - {Name} | oteviraci doba {OpenTime} - {CloseTime}";
        }
    }
}
