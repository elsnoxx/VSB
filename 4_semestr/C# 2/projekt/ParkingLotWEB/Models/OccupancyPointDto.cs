using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class OccupancyPointDto
    {
        public DateTime Time { get; set; }
        public int OccupiedSpaces { get; set; }
    }
}
