using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class ElectricCar : Car, IElectricEngine
    {
        public double BatteryCapacity { get; init; }
        public double RemainingEnergy { get; private set; }

        public ElectricCar()
        {
            RemainingEnergy = 0;
        }

        public void Charge(double quantity)
        {
            RemainingEnergy = Math.Min(RemainingEnergy + quantity, BatteryCapacity);
        }

        public override void DisplayInfo()
        {
            base.DisplayInfo();
            Console.WriteLine($"Stav baterie: {(int)(RemainingEnergy * 100 / BatteryCapacity)}%");
        }
    }
}
