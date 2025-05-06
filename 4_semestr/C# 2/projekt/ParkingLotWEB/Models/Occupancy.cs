using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class Occupancy
    {
        public int OccupancyId { get; set; }
        public int ParkingSpaceId { get; set; }
        public string LicensePlate { get; set; }
        public DateTime? StartTime { get; set; }
        public DateTime? EndTime { get; set; }
        public int? Duration { get; set; }
        public decimal? Price { get; set; }
        public string Status { get; set; }
        public int ParkingLotId { get; set; }
    }
}
