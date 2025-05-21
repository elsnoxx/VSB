using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class ParkingHistoryDto
    {
        public string licensePlate { get; set; }
        public string parkingLotName { get; set; }
        public int duration { get; set; }
        public decimal pricePerHour { get; set; }
        public decimal totalPrice { get; set; }

        public DateTime arrivalTime { get; set; }
        public DateTime? departureTime { get; set; }
    }
}
