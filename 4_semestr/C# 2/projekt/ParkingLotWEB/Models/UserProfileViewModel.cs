using System.Collections.Generic;

namespace ParkingLotWEB.Models
{
    public class UserProfileViewModel
    {
        public int Id { get; set; }
        public string Username { get; set; } = default!;
        public string FirstName { get; set; } = default!;
        public string LastName { get; set; } = default!;
        public string Email { get; set; } = default!;
        public List<Car> Cars { get; set; } = new();
    }
}