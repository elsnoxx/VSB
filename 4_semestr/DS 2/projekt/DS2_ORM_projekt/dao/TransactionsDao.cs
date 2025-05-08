using System;
using System.Data;
using System.Data.SqlClient;
using DS2_ORM_projekt.dto;

namespace DS2_ORM_projekt.dao
{
    public static class TransactionsDao
    {
        public static bool ChangeRoomAppLogic(Database db, int reservationId, int newRoomId)
        {
            bool ret = true;
            try
            {
                db.BeginTransaction();

                // 1. Zjisti termín rezervace
                var cmdGetReservation = db.CreateCommand("SELECT check_in_date, check_out_date FROM Reservation WHERE reservation_id = @id");
                cmdGetReservation.Parameters.AddWithValue("@id", reservationId);
                DateTime checkIn, checkOut;
                using (var reader = db.Select(cmdGetReservation))
                {
                    if (!reader.Read())
                    {
                        Console.WriteLine("Rezervace nenalezena");
                        db.Rollback();
                        return false;
                    }
                    checkIn = reader.GetDateTime(0);
                    checkOut = reader.GetDateTime(1);
                }

                // 2. Zkontroluj kolize
                bool available = RoomDao.IsRoomAvailable(db, newRoomId, checkIn, checkOut);
                if (!available)
                {
                    Console.WriteLine("Pokoj není v daném termínu volný.");
                    db.Rollback();
                    return false;
                }

                // 3. Zjisti typ pokoje
                int roomTypeId = RoomDao.GetRoomTypeId(db, newRoomId);

                // 4. Zjisti cenu za noc
                decimal pricePerNight = RoomTypePriceHistoryDao.GetPricePerNight(db, roomTypeId, checkIn);

                // 5. Spočítej počet nocí a cenu
                int nights = (checkOut - checkIn).Days;
                decimal newPrice = nights * pricePerNight;

                // 6. Aktualizuj rezervaci
                var cmdUpdate = db.CreateCommand("UPDATE Reservation SET room_id = @roomId, accommodation_price = @price WHERE reservation_id = @resId");
                cmdUpdate.Parameters.AddWithValue("@roomId", newRoomId);
                cmdUpdate.Parameters.AddWithValue("@price", newPrice);
                cmdUpdate.Parameters.AddWithValue("@resId", reservationId);
                db.ExecuteNonQuery(cmdUpdate);

                db.EndTransaction();
            }
            catch (Exception ex)
            {
                db.Rollback();
                Console.WriteLine("Chyba: " + ex.Message);
                ret = false;
            }
            return ret;
        }

        public static bool ChangeRoomStoredProc(Database db, int reservationId, int newRoomId)
        {
            try
            {
                db.BeginTransaction();
                var cmd = db.CreateCommand("change_room");
                cmd.CommandType = CommandType.StoredProcedure;
                cmd.Parameters.AddWithValue("@p_reservation_id", reservationId);
                cmd.Parameters.AddWithValue("@p_new_room_id", newRoomId);
                db.ExecuteNonQuery(cmd);
                db.EndTransaction();
                return true;
            }
            catch (Exception ex)
            {
                db.Rollback();
                Console.WriteLine("Chyba (SP): " + ex.Message);
                return false;
            }
        }
    }
}