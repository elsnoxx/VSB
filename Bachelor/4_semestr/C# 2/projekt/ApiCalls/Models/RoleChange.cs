using System.ComponentModel.DataAnnotations;

namespace ApiCalls.Models
{
    public class RoleChange
    {
        [Required]
        public string Role { get; set; }
    }
}