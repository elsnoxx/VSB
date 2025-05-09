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
            var cmd = db.CreateCommand(
                @"SELECT reservation_id, room_id, guest_id, employee_id, check_in_date, check_out_date, status, accommodation_price, payment_id
                  FROM Reservation WHERE reservation_id = @id");
            cmd.Parameters.AddWithValue("@id", reservationId);
            using (var reader = db.Select(cmd))
            {
                if (reader.Read())
                {
                    return new Reservation
                    {
                        ReservationId = reader.GetInt32(0),
                        RoomId = reader.GetInt32(1),
                        GuestId = reader.GetInt32(2),
                        EmployeeId = reader.GetInt32(3),
                        CheckInDate = reader.GetDateTime(4),
                        CheckOutDate = reader.GetDateTime(5),
                        Status = reader.GetString(6),
                        AccommodationPrice = reader.GetDecimal(7),
                        PaymentId = reader.GetInt32(8)
                    };
                }
            }
            return null;
        }

        public static (DateTime checkIn, DateTime checkOut)? GetReservationDates(Database db, int reservationId)
        {
            var cmd = db.CreateCommand("SELECT check_in_date, check_out_date FROM Reservation WHERE reservation_id = @id");
            cmd.Parameters.AddWithValue("@id", reservationId);
            using (var reader = db.Select(cmd))
            {
                if (reader.Read())
                {
                    return (reader.GetDateTime(0), reader.GetDateTime(1));
                }
            }
            return null;
        }

        public static void UpdateRoomAndPrice(Database db, int reservationId, int newRoomId, decimal newPrice)
        {
            var cmd = db.CreateCommand("UPDATE Reservation SET room_id = @roomId, accommodation_price = @price WHERE reservation_id = @resId");
            cmd.Parameters.AddWithValue("@roomId", newRoomId);
            cmd.Parameters.AddWithValue("@price", newPrice);
            cmd.Parameters.AddWithValue("@resId", reservationId);
            db.ExecuteNonQuery(cmd);
        }
    }
}