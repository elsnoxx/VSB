using System;
using System.Data;
using System.Data.OracleClient; // nebo Oracle.ManagedDataAccess.Client

namespace DS2_ORM_projekt.dao
{
    public class ReservationDao
    {
        private readonly string _connectionString;

        public ReservationDao(string connectionString)
        {
            _connectionString = connectionString;
        }

        // Volání uložené procedury
        public void ChangeRoomStoredProc(int reservationId, int newRoomId)
        {
            using (var conn = new OracleConnection(_connectionString))
            using (var cmd = new OracleCommand("change_room", conn))
            {
                cmd.CommandType = CommandType.StoredProcedure;
                cmd.Parameters.Add("p_reservation_id", OracleType.Number).Value = reservationId;
                cmd.Parameters.Add("p_new_room_id", OracleType.Number).Value = newRoomId;

                conn.Open();
                try
                {
                    cmd.ExecuteNonQuery();
                    Console.WriteLine("Pokoj změněn (uložená procedura).");
                }
                catch (OracleException ex)
                {
                    Console.WriteLine("Chyba: " + ex.Message);
                }
            }
        }

        // Čistě aplikační logika (vzorově)
        public void ChangeRoomAppLogic(int reservationId, int newRoomId)
        {
            using (var conn = new OracleConnection(_connectionString))
            {
                conn.Open();
                using (var tran = conn.BeginTransaction())
                {
                    try
                    {
                        // 1. Zjisti termín rezervace
                        DateTime checkIn, checkOut;
                        using (var cmd = new OracleCommand("SELECT check_in_date, check_out_date FROM Reservation WHERE reservation_id = :id", conn))
                        {
                            cmd.Parameters.Add("id", OracleType.Number).Value = reservationId;
                            using (var reader = cmd.ExecuteReader())
                            {
                                if (!reader.Read()) throw new Exception("Rezervace nenalezena");
                                checkIn = reader.GetDateTime(0);
                                checkOut = reader.GetDateTime(1);
                            }
                        }

                        // 2. Zkontroluj kolize
                        int count;
                        using (var cmd = new OracleCommand(@"SELECT COUNT(*) FROM Reservation
                            WHERE room_id = :roomId AND status != 'Cancelled'
                            AND ((:checkIn BETWEEN check_in_date AND check_out_date)
                              OR (:checkOut BETWEEN check_in_date AND check_out_date)
                              OR (check_in_date BETWEEN :checkIn AND :checkOut))", conn))
                        {
                            cmd.Parameters.Add("roomId", OracleType.Number).Value = newRoomId;
                            cmd.Parameters.Add("checkIn", OracleType.DateTime).Value = checkIn;
                            cmd.Parameters.Add("checkOut", OracleType.DateTime).Value = checkOut;
                            count = Convert.ToInt32(cmd.ExecuteScalar());
                        }
                        if (count > 0)
                        {
                            tran.Rollback();
                            Console.WriteLine("Pokoj není v daném termínu volný.");
                            return;
                        }

                        // 3. Zjisti typ pokoje
                        int roomTypeId;
                        using (var cmd = new OracleCommand("SELECT room_type_id FROM Room WHERE room_id = :id", conn))
                        {
                            cmd.Parameters.Add("id", OracleType.Number).Value = newRoomId;
                            roomTypeId = Convert.ToInt32(cmd.ExecuteScalar());
                        }

                        // 4. Zjisti cenu za noc
                        decimal pricePerNight;
                        using (var cmd = new OracleCommand(@"SELECT price_per_night FROM RoomTypePriceHistory
                            WHERE room_type_id = :rtid AND :checkIn BETWEEN valid_from AND NVL(valid_to, :checkIn)", conn))
                        {
                            cmd.Parameters.Add("rtid", OracleType.Number).Value = roomTypeId;
                            cmd.Parameters.Add("checkIn", OracleType.DateTime).Value = checkIn;
                            pricePerNight = Convert.ToDecimal(cmd.ExecuteScalar());
                        }

                        // 5. Spočítej počet nocí a cenu
                        int nights = (checkOut - checkIn).Days;
                        decimal newPrice = nights * pricePerNight;

                        // 6. Aktualizuj rezervaci
                        using (var cmd = new OracleCommand(@"UPDATE Reservation SET room_id = :roomId, accommodation_price = :price WHERE reservation_id = :resId", conn))
                        {
                            cmd.Parameters.Add("roomId", OracleType.Number).Value = newRoomId;
                            cmd.Parameters.Add("price", OracleType.Number).Value = newPrice;
                            cmd.Parameters.Add("resId", OracleType.Number).Value = reservationId;
                            cmd.ExecuteNonQuery();
                        }

                        tran.Commit();
                        Console.WriteLine("Pokoj změněn (aplikační logika).");
                    }
                    catch (Exception ex)
                    {
                        tran.Rollback();
                        Console.WriteLine("Chyba: " + ex.Message);
                    }
                }
            }
        }

        public Reservation GetReservationById(int reservationId)
        {
            using (var conn = new OracleConnection(_connectionString))
            using (var cmd = new OracleCommand("SELECT * FROM Reservation WHERE reservation_id = :id", conn))
            {
                cmd.Parameters.Add("id", OracleType.Number).Value = reservationId;
                conn.Open();
                using (var reader = cmd.ExecuteReader())
                {
                    if (reader.Read())
                    {
                        return new Reservation
                        {
                            ReservationId = reader.GetInt32(reader.GetOrdinal("reservation_id")),
                            GuestId = reader.GetInt32(reader.GetOrdinal("guest_id")),
                            RoomId = reader.GetInt32(reader.GetOrdinal("room_id")),
                            EmployeeId = reader.GetInt32(reader.GetOrdinal("employee_id")),
                            CreationDate = reader.GetDateTime(reader.GetOrdinal("creation_date")),
                            CheckInDate = reader.GetDateTime(reader.GetOrdinal("check_in_date")),
                            CheckOutDate = reader.GetDateTime(reader.GetOrdinal("check_out_date")),
                            PaymentId = reader.GetInt32(reader.GetOrdinal("payment_id")),
                            Status = reader.GetString(reader.GetOrdinal("status")),
                            AccommodationPrice = reader.GetDecimal(reader.GetOrdinal("accommodation_price"))
                        };
                    }
                }
            }
            return null;
        }

        public void UpdateRoomAndPrice(int reservationId, int newRoomId, decimal newPrice)
        {
            using (var conn = new OracleConnection(_connectionString))
            using (var cmd = new OracleCommand("UPDATE Reservation SET room_id = :roomId, accommodation_price = :price WHERE reservation_id = :resId", conn))
            {
                cmd.Parameters.Add("roomId", OracleType.Number).Value = newRoomId;
                cmd.Parameters.Add("price", OracleType.Number).Value = newPrice;
                cmd.Parameters.Add("resId", OracleType.Number).Value = reservationId;
                conn.Open();
                cmd.ExecuteNonQuery();
            }
        }
    }
}