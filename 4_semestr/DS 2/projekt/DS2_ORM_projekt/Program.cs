using System;
using DS2_ORM_projekt.dao;


namespace DS2_ORM_projekt
{
    class Program
    {
        static void Main(string[] args)
        {
            var db = new Database();
            db.Connect();

            // Příklad: změna pokoje v rámci transakce
            int reservationId = 1;
            int newRoomId = 3;
            
            var before = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine($"Před změnou: rezervace {reservationId}, nový pokoj {newRoomId}");

            bool ret = TransactionsDao.ChangeRoomAppLogic(db, reservationId, newRoomId);
            Console.WriteLine("ChangeRoomAppLogic: ret: " + ret);

            var before2 = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine($"Po změně: pokoj {before2.RoomId}, cena {before2.AccommodationPrice}");

            // Změna pokoje uloženou procedurou
            bool retSp = TransactionsDao.ChangeRoomStoredProc(db, reservationId, newRoomId);
            Console.WriteLine("ChangeRoomStoredProc: ret: " + retSp);

            var afterSp = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine($"Po změně (stored proc): pokoj {afterSp.RoomId}, cena {afterSp.AccommodationPrice}");


            db.Close();
        }
    }
}
