namespace Ukol2
{
    internal abstract class Vehicle
    {
        public string VehicleName { get; set; }
        public double AccountBalance { get; set; }

        public Vehicle() { }

        public virtual void DisplayInfo()
        {
            Console.WriteLine($"{VehicleName} | {AccountBalance} Kč");
        }

        public abstract int NumberOfWheels();
    }
}
