namespace testPripava1;

public class InvalidSpeedException : Exception
{
    public readonly double Speed;

    public InvalidSpeedException(double speed)
    {
        Speed = speed;
    }
}