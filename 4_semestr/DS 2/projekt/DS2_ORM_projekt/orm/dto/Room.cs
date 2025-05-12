using System;
namespace DS2_ORM_projekt.dto
{
    public class Room
    {
        public int RoomId { get; set; }
        public int RoomTypeId { get; set; }
        public string RoomNumber { get; set; }
        public bool IsOccupied { get; set; }
    }
}