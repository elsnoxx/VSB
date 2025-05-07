namespace ParkingLotWEB.Models
{
    public class CarCreateDto
    {
        public int UserId { get; set; }
        public string LicensePlate { get; set; } = string.Empty;
        public string BrandModel { get; set; } = string.Empty;
        public string Color { get; set; } = string.Empty;
    }
}