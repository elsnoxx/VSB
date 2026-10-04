using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class RoleChange
    {
        [Required]
        public string Role { get; set; }
    }
}