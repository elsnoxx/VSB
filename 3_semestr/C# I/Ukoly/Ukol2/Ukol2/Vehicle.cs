namespace Ukol2
{
    internal abstract class Vehicle
    {
        public string VehicleName;
        public double AccountBalance;

        public Vehicle() { }

        public virtual void DisplayInfo()
        {
            Console.WriteLine($"{VehicleName} | {AccountBalance} Kč");
        }

        public abstract int NumberOfWheels();
    }
}
