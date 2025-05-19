using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class ParkingLotProfileViewModel
    {
        public int parkingLotId { get; set; }
        public string name { get; set; }
        public decimal latitude { get; set; }
        public decimal longitude { get; set; }
        public int capacity { get; set; }
        public int freeSpaces { get; set; }
        public List<ParkingSpace> parkingSpaces { get; set; }
    }
}
