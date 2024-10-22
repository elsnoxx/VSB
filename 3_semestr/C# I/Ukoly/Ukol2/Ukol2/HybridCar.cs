using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal class HybridCar : Car, IElectricEngine, IGasolineEngine
    {
        public double FuelTankSize { get; init; }
        public double AmountOfFuel { get; set; }
        public double BatteryCapacity { get; init; }
        public double RemainingEnergy { get; set; }

        public HybridCar()
        {
            AmountOfFuel = 0;
            RemainingEnergy = 0;
        }

        public void Refuel(double quantity)
        {
            AmountOfFuel = Math.Min(AmountOfFuel + quantity, FuelTankSize);
        }

        public void Charge(double quantity)
        {
            RemainingEnergy = Math.Min(RemainingEnergy + quantity, BatteryCapacity);
        }

        public override void DisplayInfo()
        {
            base.DisplayInfo();
            Console.WriteLine($"Stav baterie: {(int)(RemainingEnergy * 100 / BatteryCapacity)}%| Stav nádrže: {(int)(AmountOfFuel * 100 / FuelTankSize)}% ");
        }

        public override int NumberOfWheels()
        {
            return 4;
        }
    }
}
