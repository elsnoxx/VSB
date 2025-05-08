using System;
using System.Data;

namespace DS2_ORM_projekt.dao
{
    public static class RoomTypePriceHistoryDao
    {
        public static decimal GetPricePerNight(Database db, int roomTypeId, DateTime checkIn)
        {
            var cmd = db.CreateCommand(@"
                SELECT price_per_night FROM RoomTypePriceHistory
                WHERE room_type_id = @rtid AND @checkIn BETWEEN valid_from AND ISNULL(valid_to, @checkIn)");
            cmd.Parameters.AddWithValue("@rtid", roomTypeId);
            cmd.Parameters.AddWithValue("@checkIn", checkIn);
            return Convert.ToDecimal(cmd.ExecuteScalar());
        }
    }
}