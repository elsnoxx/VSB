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
        public double AmountOfFuel { get; private set; }
        public double BatteryCapacity { get; init; }
        public double RemainingEnergy { get; private set; }

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
            Console.WriteLine($"Stav baterie: {Math.Round(RemainingEnergy * 100 / BatteryCapacity)}% | Stav nádrže: {Math.Round(AmountOfFuel * 100 / FuelTankSize)}%");
        }

        
    }
}
