using System.Collections.Generic;

namespace ParkingLotWEB.Models.ViewModels
{
    public class ParkingLotDetailsViewModel
    {
        public int ParkingLotId { get; set; }
        public string Name { get; set; } = string.Empty;
        public decimal Latitude { get; set; }
        public decimal Longitude { get; set; }
        public int Capacity { get; set; }
        public List<ParkingSpaceWithOwner> ParkingSpaces { get; set; } = new();
        public List<CarDto> UserCars { get; set; } = new();
    }

    public class UserCar
    {
        public string LicensePlate { get; set; }
        public string BrandModel { get; set; }
    }

    public class ParkingSpaceWithDetails
    {
        public int ParkingSpaceId { get; set; }
        public int SpaceNumber { get; set; }
        public string Status { get; set; }
        public string? OwnerId { get; set; }
    }
}