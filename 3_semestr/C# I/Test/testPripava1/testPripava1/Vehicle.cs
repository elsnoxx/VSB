namespace testPripava1;

public abstract class Vehicle
{
    
    public double Speed { get; set; }

    public Vehicle(double speed)
    {
        if (speed < 0)
        {
            throw new InvalidSpeedException(speed);
        }
        Speed = speed;
    }
}