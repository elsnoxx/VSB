using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class ReleaseRequest
    {
        public int ParkingSpaceId { get; set; }
        public int ParkingLotId { get; set; }
    }
}