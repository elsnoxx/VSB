using System;
namespace DS2_ORM_projekt.dto
{
    public class ServiceUsage
    {
        public int UsageId { get; set; }
        public int ReservationId { get; set; }
        public int ServiceId { get; set; }
        public int Quantity { get; set; }
        public decimal TotalPrice { get; set; }
    }
}