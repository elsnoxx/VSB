using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class ChargingStation
    {
        public double PricePerKWh { get; private set; }

        public ChargingStation(double price)
        {
            PricePerKWh = price;
        }

        public void Charge(IElectricEngine vehicle)
        {
            //double cost = PricePerKWh;
            //double maxCharge = Math.Min(vehicle.BatteryCapacity - vehicle.RemainingEnergy, vehicle.AccountBalance / cost);
            //vehicle.Charge(maxCharge);
            //vehicle.AccountBalance -= maxCharge * cost;
        }
    }
}
