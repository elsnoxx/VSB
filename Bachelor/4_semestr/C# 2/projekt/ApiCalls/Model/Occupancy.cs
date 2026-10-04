using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class Occupancy
    {
        public int occupancyId { get; set; }
        public int parkingSpaceId { get; set; }
        public string licensePlate { get; set; }
        public DateTime? startTime { get; set; }
        public DateTime? endTime { get; set; }
        public int? duration { get; set; }
        public decimal? price { get; set; }
        public string status { get; set; }
        public int parkingLotId { get; set; }
    }
}
