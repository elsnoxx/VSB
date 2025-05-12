using System;
namespace DS2_ORM_projekt.dto
{
    public class ServicePriceHistory
    {
        public int SphId { get; set; }
        public int ServiceId { get; set; }
        public decimal Price { get; set; }
        public DateTime ValidFrom { get; set; }
        public DateTime? ValidTo { get; set; }
    }
}