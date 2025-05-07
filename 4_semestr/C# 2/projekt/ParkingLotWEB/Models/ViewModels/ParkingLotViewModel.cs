namespace ParkingLotWEB.Models.ViewModels
{
    public class ParkingLotViewModel
    {
        public int ParkingLotId { get; set; }
        public string Name { get; set; } = string.Empty;
        public double Latitude { get; set; }
        public double Longitude { get; set; }
        public int Capacity { get; set; }
        public int FreeSpaces { get; set; }
        public List<ParkingSpaceViewModel>? ParkingSpaces { get; set; }
    }

}