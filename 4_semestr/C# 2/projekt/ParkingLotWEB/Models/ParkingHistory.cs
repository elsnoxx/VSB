namespace ParkingLotWEB.Models
{
    public class ParkingHistory
    {
        public string LicensePlate { get; set; }
        public string ParkingLotName { get; set; }
        public DateTime ArrivalTime { get; set; }
        public DateTime? DepartureTime { get; set; }
    }
}