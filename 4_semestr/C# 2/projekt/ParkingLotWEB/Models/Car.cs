using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class Car
    {
        public int CarId { get; set; }
        public int UserId { get; set; } // cizí klíč na User

        [Required]
        [StringLength(20, ErrorMessage = "Maximální délka je 20 znaků.")]
        // [RegularExpression(@"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{2,20}$", ErrorMessage = "Neplatná registrační značka.")]
        [Display(Name = "SPZ")]
        public string LicensePlate { get; set; }

        [Required]
        [StringLength(100, ErrorMessage = "Maximální délka je 100 znaků.")]
        [Display(Name = "Značka a model")]
        public string? BrandModel { get; set; }

        [Required]
        [StringLength(50, ErrorMessage = "Maximální délka je 50 znaků.")]
        [Display(Name = "Barva")]
        public string? Color { get; set; }

        public User? User { get; set; }
    }
}