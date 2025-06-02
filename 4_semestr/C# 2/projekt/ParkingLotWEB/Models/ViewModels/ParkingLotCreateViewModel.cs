using System.ComponentModel.DataAnnotations;

namespace ParkingLotWEB.Models
{
    public class ParkingLotCreateViewModel
    {
        public string Name { get; set; }
        public decimal Latitude { get; set; }
        
        public decimal Longitude { get; set; }
        [Required(ErrorMessage = "Kapacita je povinná.")]
        [Range(1, int.MaxValue, ErrorMessage = "Kapacita musí být větší než 0.")]
        public int Capacity { get; set; }
        [Range(0, int.MaxValue, ErrorMessage = "Počet volných míst nesmí být záporný.")]
        public decimal PricePerHour { get; set; }
    }
}