namespace ParkingLotWEB.Models
{
    public class ParkingSpace
    {
        public int ParkingSpaceId { get; set; }
        public int ParkingLotId { get; set; }
        public int SpaceNumber { get; set; }
        public string Status { get; set; } = string.Empty;
    }
}