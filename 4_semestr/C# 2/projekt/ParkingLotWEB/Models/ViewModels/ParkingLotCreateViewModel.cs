namespace ParkingLotWEB.Models
{
    public class ParkingLotCreateViewModel
    {
        public string Name { get; set; }
        public decimal Latitude { get; set; }
        public decimal Longitude { get; set; }
        public int Capacity { get; set; }
    }
}