using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class ParkingSpace
    {
        public int parkingSpaceId { get; set; }
        public int parkingLotId { get; set; }
        public int spaceNumber { get; set; }
        public string status { get; set; } = string.Empty;
    }
}
