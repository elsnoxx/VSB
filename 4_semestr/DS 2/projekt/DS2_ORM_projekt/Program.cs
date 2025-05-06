using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DS2_ORM_projekt
{
    internal class Program
    {
        static void Main(string[] args)
        {
            string connStr = "YOUR_ORACLE_CONNECTION_STRING";
            var reservationDao = new ReservationDao(connStr);

            // Test volání uložené procedury
            reservationDao.ChangeRoomStoredProc(1, 2);

            // Test aplikační logiky
            reservationDao.ChangeRoomAppLogic(1, 3);
        }
    }
}
