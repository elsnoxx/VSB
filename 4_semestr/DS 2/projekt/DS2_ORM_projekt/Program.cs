using DS2_ORM_projekt.dao;
using DS2_ORM_projekt.dto;
using System;

namespace DS2_ORM_projekt
{
    class Program
    {
        static void Main(string[] args)
        {
            var db = new Database();
            db.Connect();

            Console.WriteLine("========== DEMONSTRACE ZMĚNY POKOJE V REZERVACI ==========");

            int reservationId = 1;

            Random random = new Random();
            int newRoomId = random.Next(1, 21);

            Console.WriteLine("\n--- 1. STAV PŘED ZMĚNOU ---");
            var before = ReservationDao.GetReservationById(db, reservationId);
            PrintReservation(before);

            Console.WriteLine($"\n--- 2. POKUS O ZMĚNU POKOJE NA {newRoomId} (aplikační logika) ---");
            bool ret = TransactionsDao.ChangeRoomAppLogic(db, reservationId, newRoomId);
            Console.WriteLine("Výsledek změny (AppLogic): " + (ret ? "ÚSPĚCH" : "NEÚSPĚCH"));

            Console.WriteLine("\n--- 3. STAV PO ZMĚNĚ (aplikační logika) ---");
            var after = ReservationDao.GetReservationById(db, reservationId);
            PrintReservation(after);

            if (before != null && after != null)
            {
                Console.WriteLine($"\nZměna pokoje: {before.RoomId} -> {after.RoomId}");
            }

            Console.WriteLine($"\n--- 4. POKUS O ZMĚNU POKOJE NA {newRoomId} (uložená procedura) ---");
            newRoomId = random.Next(1, 21);
            bool retSp = TransactionsDao.ChangeRoomStoredProc(db, reservationId, newRoomId);
            Console.WriteLine("Výsledek změny (StoredProc): " + (retSp ? "ÚSPĚCH" : "NEÚSPĚCH"));

            Console.WriteLine("\n--- 5. STAV PO ZMĚNĚ (uložená procedura) ---");
            var afterSp = ReservationDao.GetReservationById(db, reservationId);
            PrintReservation(afterSp);

            if (after != null && afterSp != null)
            {
                Console.WriteLine($"\nZměna pokoje (stored proc): {after.RoomId} -> {afterSp.RoomId}");
            }

            db.Close();
        }

        static void PrintReservation(Reservation res, string prefix = "")
        {
            if (res == null)
            {
                Console.WriteLine($"{prefix}Rezervace nenalezena.");
                return;
            }
            Console.WriteLine($"{prefix}Rezervace ID: {res.ReservationId}");
            Console.WriteLine($"  Pokoj: {res.RoomId}, Host: {res.GuestId}, Zaměstnanec: {res.EmployeeId}, Check-in: {res.CheckInDate:yyyy-MM-dd}, Check-out: {res.CheckOutDate:yyyy-MM-dd}, Stav: {res.Status}, Cena: {res.AccommodationPrice:C},  PaymentId: {res.PaymentId}");
        }
    }
}
