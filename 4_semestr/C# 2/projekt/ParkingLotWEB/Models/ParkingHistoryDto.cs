namespace ParkingLotWEB.Models
{
    public class ParkingHistoryDto
    {
        public string LicensePlate { get; set; }
        public string ParkingLotName { get; set; }
        public DateTime ArrivalTime { get; set; }
        public DateTime? DepartureTime { get; set; }
    }
}