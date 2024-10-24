namespace Ukol2
{
    internal class FillingStation
    {
        public double PricePerLiter { get; init; }

        public FillingStation(double price)
        {
            PricePerLiter = price ;
        }

        public void Refuel(Vehicle vehicle)
        {
            if (vehicle is IGasolineEngine gasolineVehicle)
            {
                double costPerLiter = PricePerLiter + 4 * vehicle.NumberOfWheels();
                double maxFuel = Math.Min(gasolineVehicle.FuelTankSize - gasolineVehicle.AmountOfFuel, vehicle.AccountBalance / costPerLiter);
                gasolineVehicle.Refuel(maxFuel);
                vehicle.AccountBalance = Math.Round(vehicle.AccountBalance - maxFuel * costPerLiter, 2); ;
            }
        }
    }
}
