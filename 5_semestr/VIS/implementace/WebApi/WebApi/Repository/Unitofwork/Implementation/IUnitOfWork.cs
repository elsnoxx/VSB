using System.Data;

namespace WebApi.Repository.Unitofwork.Implementation
{
    public interface IUnitOfWork : IAsyncDisposable
    {
        IDbConnection Connection { get; }
        IDbTransaction? Transaction { get; }

        Task OpenAsync(CancellationToken ct = default);
        Task BeginAsync(CancellationToken ct = default);
        Task CommitAsync(CancellationToken ct = default);
        Task RollbackAsync(CancellationToken ct = default);
    }
}
