namespace WebApi.Models.Domain
{
    public class Location
    {
        public Guid Id { get; }
        public string Name { get; private set; }
        public Guid? ParentId { get; private set; }
        public DateTime CreatedAtUtc { get; }

        public Location(Guid id, string name, Guid? parentId, DateTime createdAtUtc)
        {
            Id = id;
            Name = name;
            ParentId = parentId;
            CreatedAtUtc = createdAtUtc;
        }

        public void Rename(string name) => Name = name;
        public void ChangeParent(Guid? parentId) => ParentId = parentId;
    }
}
