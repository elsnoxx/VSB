namespace ParkingLotWEB.Models
{
    public class StatusHistory
    {
        public int id { get; set; }
        public int ParkingSpaceId { get; set; }
        public int SpaceNumber { get; set; }
        public string Status { get; set; }
        public int ParkingLotId { get; set; } 
        public DateTime ChangeTime { get; set; }
    }
}