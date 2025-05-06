namespace ParkingLotWEB.Models
{
    public class StatusHistory
    {
        public int Id { get; set; }
        public int ParkingSpaceId { get; set; }
        public string Status { get; set; }
        public DateTime ChangeTime { get; set; }
    }
}