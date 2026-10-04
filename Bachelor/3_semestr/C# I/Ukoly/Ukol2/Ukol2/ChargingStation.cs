using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class ChargingStation
    {
        public double PricePerKWh { get; set; }

        public ChargingStation(double price)
        {
            PricePerKWh = price;
        }

        public void Charge(Vehicle vehicle)
        {
            if (vehicle is IElectricEngine electricVehicle)
            {
                double costPerKWh = PricePerKWh + 4 * vehicle.NumberOfWheels();
                double maxCharge = Math.Min(electricVehicle.BatteryCapacity - electricVehicle.RemainingEnergy, vehicle.AccountBalance / costPerKWh);
                electricVehicle.Charge(maxCharge);
                vehicle.AccountBalance = Math.Round(vehicle.AccountBalance - maxCharge * costPerKWh, 2);
            }
        }
    }
}
