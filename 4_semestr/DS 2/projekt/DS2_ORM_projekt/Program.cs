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

            bool running = true;
            while (running)
            {
                Console.Clear();
                Console.WriteLine("========== HOTELOVÝ SYSTÉM - SPRÁVA POKOJŮ ==========");
                Console.WriteLine("1. Změnit pokoj pomocí aplikační logiky");
                Console.WriteLine("2. Změnit pokoj pomocí uložené procedury");
                Console.WriteLine("3. Zkontrolovat dostupnost pokoje");
                Console.WriteLine("4. Zobrazit informace o rezervaci");
                Console.WriteLine("0. Konec");
                Console.Write("\nVyberte možnost: ");

                if (int.TryParse(Console.ReadLine(), out int choice))
                {
                    switch (choice)
                    {
                        case 1:
                            ChangeRoomUsingAppLogic(db);
                            break;
                        case 2:
                            ChangeRoomUsingStoredProcedure(db);
                            break;
                        case 3:
                            CheckRoomAvailability(db);
                            break;
                        case 4:
                            DisplayReservationInfo(db);
                            break;
                        case 0:
                            running = false;
                            break;
                        default:
                            Console.WriteLine("Neplatná volba.");
                            break;
                    }
                }
                else
                {
                    Console.WriteLine("Neplatný vstup. Zadejte číslo.");
                }

                if (running)
                {
                    Console.WriteLine("\nStiskněte libovolnou klávesu pro pokračování...");
                    Console.ReadKey();
                }
            }

            db.Close();
            Console.WriteLine("Program byl ukončen.");
        }

        static void ChangeRoomUsingAppLogic(Database db)
        {
            Console.WriteLine("\n--- ZMĚNA POKOJE POMOCÍ APLIKAČNÍ LOGIKY ---");
            
            Console.Write("Zadejte ID rezervace: ");
            if (!int.TryParse(Console.ReadLine(), out int reservationId))
            {
                Console.WriteLine("Neplatné ID rezervace.");
                return;
            }

            var before = ReservationDao.GetReservationById(db, reservationId);
            if (before == null)
            {
                Console.WriteLine("Rezervace s tímto ID nebyla nalezena.");
                return;
            }

            Console.WriteLine("\nAktuální stav rezervace:");
            PrintReservation(before);

            Console.Write("\nZadejte ID nového pokoje: ");
            if (!int.TryParse(Console.ReadLine(), out int newRoomId))
            {
                Console.WriteLine("Neplatné ID pokoje.");
                return;
            }

            Console.WriteLine($"\nPokus o změnu pokoje na {newRoomId}...");
            bool result = TransactionsDao.ChangeRoomAppLogic(db, reservationId, newRoomId);
            Console.WriteLine("Výsledek: " + (result ? "ÚSPĚCH" : "NEÚSPĚCH"));

            var after = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine("\nStav po změně:");
            PrintReservation(after);

            if (before != null && after != null)
            {
                Console.WriteLine($"\nZměna pokoje: {before.RoomId} -> {after.RoomId}");
            }
        }

        static void ChangeRoomUsingStoredProcedure(Database db)
        {
            Console.WriteLine("\n--- ZMĚNA POKOJE POMOCÍ ULOŽENÉ PROCEDURY ---");
            
            Console.Write("Zadejte ID rezervace: ");
            if (!int.TryParse(Console.ReadLine(), out int reservationId))
            {
                Console.WriteLine("Neplatné ID rezervace.");
                return;
            }

            var before = ReservationDao.GetReservationById(db, reservationId);
            if (before == null)
            {
                Console.WriteLine("Rezervace s tímto ID nebyla nalezena.");
                return;
            }

            Console.WriteLine("\nAktuální stav rezervace:");
            PrintReservation(before);

            Console.Write("\nZadejte ID nového pokoje: ");
            if (!int.TryParse(Console.ReadLine(), out int newRoomId))
            {
                Console.WriteLine("Neplatné ID pokoje.");
                return;
            }

            Console.WriteLine($"\nPokus o změnu pokoje na {newRoomId}...");
            bool result = TransactionsDao.ChangeRoomStoredProc(db, reservationId, newRoomId);
            Console.WriteLine("Výsledek: " + (result ? "ÚSPĚCH" : "NEÚSPĚCH"));

            var after = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine("\nStav po změně:");
            PrintReservation(after);

            if (before != null && after != null)
            {
                Console.WriteLine($"\nZměna pokoje: {before.RoomId} -> {after.RoomId}");
            }
        }

        static void CheckRoomAvailability(Database db)
        {
            Console.WriteLine("\n--- KONTROLA DOSTUPNOSTI POKOJE ---");
            
            Console.Write("Zadejte ID pokoje: ");
            if (!int.TryParse(Console.ReadLine(), out int roomId))
            {
                Console.WriteLine("Neplatné ID pokoje.");
                return;
            }

            Console.Write("Zadejte datum check-in (YYYY-MM-DD): ");
            if (!DateTime.TryParse(Console.ReadLine(), out DateTime checkIn))
            {
                Console.WriteLine("Neplatné datum check-in.");
                return;
            }

            Console.Write("Zadejte datum check-out (YYYY-MM-DD): ");
            if (!DateTime.TryParse(Console.ReadLine(), out DateTime checkOut))
            {
                Console.WriteLine("Neplatné datum check-out.");
                return;
            }

            bool isAvailable = RoomDao.IsRoomAvailable(db, roomId, checkIn, checkOut);
            
            Console.WriteLine($"\nPokoj {roomId} je v období {checkIn:yyyy-MM-dd} až {checkOut:yyyy-MM-dd} " + 
                (isAvailable ? "DOSTUPNÝ" : "OBSAZENÝ"));
        }

        static void DisplayReservationInfo(Database db)
        {
            Console.WriteLine("\n--- INFORMACE O REZERVACI ---");
            
            Console.Write("Zadejte ID rezervace: ");
            if (!int.TryParse(Console.ReadLine(), out int reservationId))
            {
                Console.WriteLine("Neplatné ID rezervace.");
                return;
            }

            var reservation = ReservationDao.GetReservationById(db, reservationId);
            Console.WriteLine();
            PrintReservation(reservation);
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
