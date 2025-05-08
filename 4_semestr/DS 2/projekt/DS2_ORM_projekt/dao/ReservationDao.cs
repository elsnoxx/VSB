using System;
using System.Data;
using System.Data.SqlClient;
using DS2_ORM_projekt.dto;


namespace DS2_ORM_projekt.dao
{
    public static class ReservationDao
    {
        public static Reservation GetReservationById(Database db, int reservationId)
        {
            var cmd = db.CreateCommand("SELECT reservation_id, room_id, accommodation_price FROM Reservation WHERE reservation_id = @id");
            cmd.Parameters.AddWithValue("@id", reservationId);
            using (var reader = db.Select(cmd))
            {
                if (reader.Read())
                {
                    return new Reservation
                    {
                        ReservationId = reader.GetInt32(0),
                        RoomId = reader.GetInt32(1),
                        AccommodationPrice = reader.GetDecimal(2)
                    };
                }
            }
            return null;
        }
    }
}