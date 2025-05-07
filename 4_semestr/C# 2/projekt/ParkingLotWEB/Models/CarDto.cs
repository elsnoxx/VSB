namespace ParkingLotWEB.Models
{
    public class CarDto
    {
        public int CarId { get; set; }
        public int UserId { get; set; }
        public string LicensePlate { get; set; } = string.Empty;
        public string BrandModel { get; set; } = string.Empty;
        public string Color { get; set; } = string.Empty;
    }

}