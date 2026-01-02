namespace WebApi.Models.DB
{
    public class LocationRow
    {
        public Guid Id { get; set; }
        public string Name { get; set; } = null!;
        public Guid? ParentId { get; set; }
        public DateTime CreatedAtUtc { get; set; }
    }
}
