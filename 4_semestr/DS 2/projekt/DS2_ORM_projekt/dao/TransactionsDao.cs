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
            try
            {
                db.BeginTransaction();

                var dates = ReservationDao.GetReservationDates(db, reservationId);
                if (dates == null)
                {
                    Console.WriteLine("Rezervace nenalezena");
                    db.Rollback();
                    return false;
                }

                DateTime checkIn = dates.Value.checkIn;
                DateTime checkOut = dates.Value.checkOut;

                if (!RoomDao.IsRoomAvailable(db, newRoomId, checkIn, checkOut))
                {
                    Console.WriteLine("Pokoj není v daném termínu volný.");
                    db.Rollback();
                    return false;
                }

                int roomTypeId = RoomDao.GetRoomTypeId(db, newRoomId);
                decimal pricePerNight = RoomTypePriceHistoryDao.GetPricePerNight(db, roomTypeId, checkIn);
                int nights = (checkOut - checkIn).Days;
                decimal newPrice = nights * pricePerNight;

                ReservationDao.UpdateRoomAndPrice(db, reservationId, newRoomId, newPrice);

                db.EndTransaction();
                return true;
            }
            catch (Exception ex)
            {
                db.Rollback();
                Console.WriteLine("Chyba: " + ex.Message);
                return false;
            }
        }

        public static bool ChangeRoomStoredProc(Database db, int reservationId, int newRoomId)
        {
            try
            {
                db.BeginTransaction();
                var cmd = db.CreateCommand("change_room");
                cmd.CommandType = CommandType.StoredProcedure;
                cmd.Parameters.AddWithValue("@reservation_id", reservationId);
                cmd.Parameters.AddWithValue("@new_room_id", newRoomId);
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