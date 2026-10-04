using System;

namespace DS2_ORM_projekt.dto
{
    public class Reservation
    {
        public int ReservationId { get; set; }
        public int GuestId { get; set; }
        public int RoomId { get; set; }
        public int EmployeeId { get; set; }
        public DateTime CreationDate { get; set; }
        public DateTime CheckInDate { get; set; }
        public DateTime CheckOutDate { get; set; }
        public int PaymentId { get; set; }
        public string Status { get; set; }
        public decimal AccommodationPrice { get; set; }
    }
}