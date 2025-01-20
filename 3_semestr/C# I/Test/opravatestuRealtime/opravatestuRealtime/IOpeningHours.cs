using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace opravatestuRealtime
{
    public interface IOpeningHours
    {
        
        public int? OpenTime { get; set; }

        public int? CloseTime { get; set; }

    }
}
