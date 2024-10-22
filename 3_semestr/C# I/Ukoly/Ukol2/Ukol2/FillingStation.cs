using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class FillingStation
    {
        public double PricePerLiter { get; init; }

        public FillingStation(double price)
        {
            PricePerLiter = price + 4;
        }

        public void Refuel(IGasolineEngine vehicle)
        {
            //double cost = PricePerLiter * vehicle.NumberOfWheels();
            //double maxFuel = Math.Min(vehicle.FuelTankSize - vehicle.FuelLevel, vehicle.AccountBalance / cost);
            //vehicle.Refuel(maxFuel);
            //vehicle.AccountBalance -= maxFuel * cost;
        }
    }
}
