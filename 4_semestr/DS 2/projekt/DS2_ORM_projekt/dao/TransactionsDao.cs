using System;
using System.Data.OracleClient;
using DS2_ORM_projekt.dto;

namespace DS2_ORM_projekt.dao
{
    public class TransactionsDao
    {
        private readonly ReservationDao _reservationDao;
        private readonly RoomDao _roomDao;
        private readonly RoomTypePriceHistoryDao _priceHistoryDao;

        public TransactionsDao(string connectionString)
        {
            _reservationDao = new ReservationDao(connectionString);
            _roomDao = new RoomDao(connectionString);
            _priceHistoryDao = new RoomTypePriceHistoryDao(connectionString);
        }

        public void ChangeRoomAppLogic(int reservationId, int newRoomId)
        {
            // 1. Zjisti termín rezervace
            var reservation = _reservationDao.GetReservationById(reservationId);
            if (reservation == null)
            {
                Console.WriteLine("Rezervace nenalezena");
                return;
            }

            DateTime checkIn = reservation.CheckInDate;
            DateTime checkOut = reservation.CheckOutDate;

            // 2. Zkontroluj kolize (implementujte obdobně v RoomDao)
            // ...

            // 3. Zjisti typ pokoje
            int roomTypeId = _roomDao.GetRoomTypeId(newRoomId);

            // 4. Zjisti cenu za noc
            decimal pricePerNight = _priceHistoryDao.GetPricePerNight(roomTypeId, checkIn);

            // 5. Spočítej počet nocí a cenu
            int nights = (checkOut - checkIn).Days;
            decimal newPrice = nights * pricePerNight;

            // 6. Aktualizuj rezervaci
            _reservationDao.UpdateRoomAndPrice(reservationId, newRoomId, newPrice);

            Console.WriteLine("Pokoj změněn (aplikační logika, rozděleno do vrstev).");
        }
    }
}