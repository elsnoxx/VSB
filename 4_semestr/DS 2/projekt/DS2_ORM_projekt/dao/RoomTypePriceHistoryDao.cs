public decimal GetPricePerNight(int roomTypeId, DateTime checkIn)
{
    using (var conn = new OracleConnection(_connectionString))
    using (var cmd = new OracleCommand(@"SELECT price_per_night FROM RoomTypePriceHistory
        WHERE room_type_id = :rtid AND :checkIn BETWEEN valid_from AND NVL(valid_to, :checkIn)", conn))
    {
        cmd.Parameters.Add("rtid", OracleType.Number).Value = roomTypeId;
        cmd.Parameters.Add("checkIn", OracleType.DateTime).Value = checkIn;
        conn.Open();
        return Convert.ToDecimal(cmd.ExecuteScalar());
    }
}