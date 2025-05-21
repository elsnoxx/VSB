namespace ParkingLotWEB.Models.Entities
{
    public class ParkingLotDto
    {
        public int ParkingLotId { get; set; }
        public string Name { get; set; }
        public decimal Latitude { get; set; }
        public decimal Longitude { get; set; }
        public decimal PricePerHour { get; set; }
        public int Capacity { get; set; }
        public int FreeSpaces { get; set; }

        public List<ParkingSpaceDto> ParkingSpaces { get; set; } = new();
    }
    public class ParkingSpaceDto
    {
        public int ParkingSpaceId { get; set; }
        public int SpaceNumber { get; set; }
        public int Number { get; set; }
        public bool IsOccupied { get; set; }
        public int ParkingLotId { get; set; }
        public string Status { get; set; } = string.Empty;
    }
}