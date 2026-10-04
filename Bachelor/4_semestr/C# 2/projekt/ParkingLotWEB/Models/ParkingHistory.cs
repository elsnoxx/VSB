namespace ParkingLotWEB.Models
{
    public class ParkingHistory
    {
        public string LicensePlate { get; set; }
        public string ParkingLotName { get; set; }
        public int duration { get; set; }
        public decimal PricePerHour { get; set; }
        public decimal TotalPrice { get; set; }
        public DateTime ArrivalTime { get; set; }
        public DateTime? DepartureTime { get; set; }
    }
}