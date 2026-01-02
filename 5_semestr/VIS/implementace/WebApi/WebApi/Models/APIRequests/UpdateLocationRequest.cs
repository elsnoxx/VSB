namespace WebApi.Models.APIRequests
{
    public class UpdateLocationRequest
    {
        public string Name { get; set; } = null!;
        public Guid? ParentId { get; set; }
    }
}
