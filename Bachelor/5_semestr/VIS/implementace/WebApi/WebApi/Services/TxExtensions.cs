using WebApi.Repository.Unitofwork.Implementation;

namespace WebApi.Services
{
    public static class TxExtensions
    {
        public static async Task<T> ExecuteInTransactionAsync<T>(
            this IUnitOfWork uow,
            Func<Task<T>> action,
            CancellationToken ct = default)
        {
            await uow.BeginAsync(ct);
            try
            {
                var result = await action();
                await uow.CommitAsync(ct);
                return result;
            }
            catch
            {
                await uow.RollbackAsync(ct);
                throw;
            }
            finally
            {
                await uow.DisposeAsync();
            }
        }

        public static async Task ExecuteInTransactionAsync(
            this IUnitOfWork uow,
            Func<Task> action,
            CancellationToken ct = default)
        {
            await uow.BeginAsync(ct);
            try
            {
                await action();
                await uow.CommitAsync(ct);
            }
            catch
            {
                await uow.RollbackAsync(ct);
                throw;
            }
            finally
            {
                await uow.DisposeAsync();
            }
        }
    }
}
