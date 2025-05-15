using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class CurrentParkingDto
    {
        public string licensePlate { get; set; }
        public int parkingLotId { get; set; }
        public string parkingLotName { get; set; }
        public DateTime arrivalTime { get; set; }
        public int parkingSpaceId { get; set; }

    }
}
