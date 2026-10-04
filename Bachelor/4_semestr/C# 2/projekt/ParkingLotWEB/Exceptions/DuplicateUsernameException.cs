namespace ParkingLotWEB.Exceptions
{
    public class DuplicateUsernameException : Exception
    {
        public DuplicateUsernameException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
    }
}
