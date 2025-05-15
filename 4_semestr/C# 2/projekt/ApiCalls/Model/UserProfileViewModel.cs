using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class UserProfileViewModel
    {
        public int id { get; set; }
        public string username { get; set; }
        public string firstName { get; set; }
        public string lastName { get; set; }
        public string email { get; set; }
        public string role { get; set; }
        public string password { get; set; }
        public ObservableCollection<CarDto> cars { get; set; }
        public ObservableCollection<ParkingHistoryDto> parkingHistory { get; set; }
        public ObservableCollection<CurrentParkingDto> currentParking { get; set; }
    }
}
