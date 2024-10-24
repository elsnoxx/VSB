using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class GasolineCar : Car, IGasolineEngine
    {
        public double FuelTankSize { get; init; }
        public double AmountOfFuel { get; set; }

        public GasolineCar()
        {
            AmountOfFuel = 0;
        }

        public void Refuel(double quantity)
        {
            AmountOfFuel = Math.Min(AmountOfFuel + quantity, FuelTankSize);
        }

        public override void DisplayInfo()
        {
            base.DisplayInfo();
            Console.WriteLine($"Stav nádrže: {(int)(AmountOfFuel * 100 / FuelTankSize)}%");
        }

    }
}
