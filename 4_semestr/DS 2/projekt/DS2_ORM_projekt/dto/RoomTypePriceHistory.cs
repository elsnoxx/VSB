namespace DS2_ORM_projekt.dto
{
    public class RoomTypePriceHistory
    {
        public int RtphId { get; set; }
        public int RoomTypeId { get; set; }
        public decimal PricePerNight { get; set; }
        public DateTime ValidFrom { get; set; }
        public DateTime? ValidTo { get; set; }
    }
}