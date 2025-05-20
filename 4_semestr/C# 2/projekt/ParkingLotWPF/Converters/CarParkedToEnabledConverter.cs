using System;
using System.Globalization;
using System.Linq;
using System.Windows.Data;
using ApiCalls.Model;

namespace ParkingLotWPF.Converters
{
    public class CarParkedToEnabledConverter : IMultiValueConverter
    {
        public object Convert(object[] values, Type targetType, object parameter, CultureInfo culture)
        {
            var car = values[0] as CarDto;
            var currentParking = values[1] as System.Collections.IEnumerable;
            if (car == null || currentParking == null)
                return true;

            foreach (var item in currentParking)
            {
                if (item is CurrentParkingDto parking && parking.licensePlate == car.licensePlate)
                    return false;
            }
            return true;
        }

        public object[] ConvertBack(object value, Type[] targetTypes, object parameter, CultureInfo culture)
        {
            throw new NotImplementedException();
        }
    }
}