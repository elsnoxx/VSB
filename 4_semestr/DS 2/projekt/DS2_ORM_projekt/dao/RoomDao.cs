public int GetRoomTypeId(int roomId)
{
    using (var conn = new OracleConnection(_connectionString))
    using (var cmd = new OracleCommand("SELECT room_type_id FROM Room WHERE room_id = :id", conn))
    {
        cmd.Parameters.Add("id", OracleType.Number).Value = roomId;
        conn.Open();
        return Convert.ToInt32(cmd.ExecuteScalar());
    }
}