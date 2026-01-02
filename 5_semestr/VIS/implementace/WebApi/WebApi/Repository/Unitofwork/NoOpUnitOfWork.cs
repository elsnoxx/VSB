using System.Data;
using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Repository.Unitofwork
{
    public class NoOpUnitOfWork : IUnitOfWork
    {
        public IDbConnection Connection => throw new NotSupportedException("No DB connection in InMemory mode.");
        public IDbTransaction? Transaction => null;

        public Task OpenAsync(CancellationToken ct = default) => Task.CompletedTask;
        public Task BeginAsync(CancellationToken ct = default) => Task.CompletedTask;
        public Task CommitAsync(CancellationToken ct = default) => Task.CompletedTask;
        public Task RollbackAsync(CancellationToken ct = default) => Task.CompletedTask;

        public ValueTask DisposeAsync() => ValueTask.CompletedTask;
    }
}
