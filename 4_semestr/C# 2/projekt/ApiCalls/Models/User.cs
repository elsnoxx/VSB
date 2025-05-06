using System.ComponentModel.DataAnnotations;

namespace ApiCalls.Models
{
    public class User
    {
        public int Id { get; set; }

        [Required]
        [StringLength(50)]
        public string Username { get; set; } = default!;

        [Required]
        [StringLength(100)]
        public string Password { get; set; } = default!;

        [Required]
        public string Role { get; set; } = default!;
    }
}