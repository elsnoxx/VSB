using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class ResetPassword
    {
        public int Id { get; set; }
        [Required]
        [StringLength(100, MinimumLength = 8, ErrorMessage = "Heslo musí mít alespoň 8 znaků.")]
        [DataType(DataType.Password)]
        [RegularExpression(@"^(?=.*[A-Z])(?=.*\d)(?=.*[!@#$%^&*]).+$", 
            ErrorMessage = "Heslo musí obsahovat alespoň jedno velké písmeno, číslici a speciální znak.")]
        public string Password { get; set; } = default!;

        [Required]
        [DataType(DataType.Password)]
        [Compare("Password", ErrorMessage = "Hesla se neshodují.")]
        public string ConfirmPassword { get; set; } = default!;
    }
}