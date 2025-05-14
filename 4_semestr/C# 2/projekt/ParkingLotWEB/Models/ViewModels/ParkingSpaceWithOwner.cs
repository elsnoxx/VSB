namespace ParkingLotWEB.Models
{
    public class ParkingSpaceWithOwner
    {
        public int ParkingSpaceId { get; set; }
        public int ParkingLotId { get; set; }
        public int SpaceNumber { get; set; }
        public string Status { get; set; }
        public string? OwnerId { get; set; }
        public int? CarId { get; set; }
    }
}