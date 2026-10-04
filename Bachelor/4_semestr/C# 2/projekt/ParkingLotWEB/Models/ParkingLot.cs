namespace ParkingLotWEB.Models
{
    public class ParkingLot
    {
        public int ParkingLotId { get; set; }
        public string Name { get; set; }
        public decimal Latitude { get; set; }
        public decimal Longitude { get; set; }

        public decimal PricePerHour { get; set; }
        public int Capacity { get; set; }
        public int FreeSpaces { get; set; }
        public List<ParkingSpace>? ParkingSpaces { get; set; }

    }
}
