namespace WebApi.Models.APIRequests
{
    public class CreateLocationRequest
    {
        public string Name { get; set; } = null!;
        public Guid? ParentId { get; set; }
    }
}
