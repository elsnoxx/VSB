namespace ParkingLotWEB.Models
{
    public class ParkingSpaceDetails
    {
        public int ParkingSpaceId { get; set; }
        public int ParkingLotId { get; set; }
        public int SpaceNumber { get; set; }
        public string Status { get; set; } = string.Empty;
        public string? LicensePlate { get; set; }
        public DateTime? StartTime { get; set; }
        public DateTime? EndTime { get; set; }
    }
}