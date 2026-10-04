using System;
namespace DS2_ORM_projekt.dto
{
    public class Payment
    {
        public int PaymentId { get; set; }
        public decimal TotalAccommodation { get; set; }
        public decimal TotalExpenses { get; set; }
        public DateTime? PaymentDate { get; set; }
        public bool IsPaid { get; set; }
    }
}