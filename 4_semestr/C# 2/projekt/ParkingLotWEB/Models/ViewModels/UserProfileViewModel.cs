using System.Collections.Generic;

namespace ParkingLotWEB.Models
{
    public class UserProfileViewModel
    {
        public int Id { get; set; }
        public string Username { get; set; }
        public required string FirstName { get; set; }
        public required string LastName { get; set; }
        public required string Email { get; set; }
        public List<CarDto> Cars { get; set; } = new();

        // Přidání historie parkování
        public List<ParkingHistoryDto> ParkingHistory { get; set; } = new();

        // Přidání aktuálně zaparkovaných aut
        public List<CurrentParkingDto> CurrentParking { get; set; } = new();
    }
}