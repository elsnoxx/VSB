namespace DS2_ORM_projekt.dto
{
    public class Guest
    {
        public int GuestId { get; set; }
        public string Firstname { get; set; }
        public string Lastname { get; set; }
        public string Email { get; set; }
        public string? Phone { get; set; }
        public DateTime BirthDate { get; set; }
        public string Street { get; set; }
        public string City { get; set; }
        public string PostalCode { get; set; }
        public string Country { get; set; }
        public string GuestType { get; set; }
        public DateTime RegistrationDate { get; set; }
        public string? Notes { get; set; }
    }
}