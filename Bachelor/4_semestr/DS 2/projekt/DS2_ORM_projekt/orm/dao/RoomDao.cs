using System;
using System.Data;

namespace DS2_ORM_projekt.dao
{
    public static class RoomDao
    {
        public static int GetRoomTypeId(Database db, int roomId)
        {
            var cmd = db.CreateCommand("SELECT room_type_id FROM Room WHERE room_id = @id");
            cmd.Parameters.AddWithValue("@id", roomId);
            return Convert.ToInt32(cmd.ExecuteScalar());
        }

        public static bool IsRoomAvailable(Database db, int roomId, DateTime checkIn, DateTime checkOut)
        {
            var cmd = db.CreateCommand(@"
                SELECT COUNT(*) FROM Reservation
                WHERE room_id = @roomId AND status != 'Cancelled'
                AND ((@checkIn BETWEEN check_in_date AND check_out_date)
                  OR (@checkOut BETWEEN check_in_date AND check_out_date)
                  OR (check_in_date BETWEEN @checkIn AND @checkOut))");
            cmd.Parameters.AddWithValue("@roomId", roomId);
            cmd.Parameters.AddWithValue("@checkIn", checkIn);
            cmd.Parameters.AddWithValue("@checkOut", checkOut);
            int count = Convert.ToInt32(cmd.ExecuteScalar());
            return count == 0;
        }
    }
}