namespace ParkingLotWEB.Models
{
    public class CurrentParkingDto
    {
        public string LicensePlate { get; set; }
        public string ParkingLotName { get; set; }
        public DateTime ArrivalTime { get; set; }
        public int ParkingSpaceId { get; set; }

    }
}