using Dapper;
using MySql.Data.MySqlClient;
using ParkingLotWEB.Models;


namespace ParkingLotWEB.Database
{
    public class ParkingSpaceRepository
    {
        private readonly DapperRepository _repository;

        public ParkingSpaceRepository(IConfiguration configuration)
        {
            _repository = new DapperRepository(configuration);
        }

        // Získání všech parkovacích míst pro parkoviště
        public async Task<IEnumerable<ParkingSpace>> GetByParkingLotIdAsync(int parkingLotId)
        {
            string sql = @"
            SELECT * 
            FROM ParkingSpace 
            WHERE parking_lot_id = @ParkingLotId";

            return await _repository.QueryAsync<ParkingSpace>(sql, new { ParkingLotId = parkingLotId });
        }

        // Získání dostupných parkovacích míst
        public async Task<IEnumerable<ParkingSpace>> GetAvailableSpacesAsync(int parkingLotId)
        {
            string sql = @"
            SELECT * 
            FROM ParkingSpace 
            WHERE parking_lot_id = @ParkingLotId 
            AND status = 'available'";

            return await _repository.QueryAsync<ParkingSpace>(sql, new { ParkingLotId = parkingLotId });
        }

        // Aktualizace stavu parkovacího místa a záznam historie
        public async Task<bool> UpdateStatusAsync(int parkingSpaceId, string newStatus)
        {
            using (var connection = new MySqlConnection(_repository._connectionString))
            {
                await connection.OpenAsync();

                // Použití transakce pro zajištění atomicity obou operací
                using (var transaction = connection.BeginTransaction())
                {
                    try
                    {
                        // Aktualizace stavu parkovacího místa
                        string updateSql = @"
                        UPDATE ParkingSpace 
                        SET status = @Status 
                        WHERE parking_space_id = @ParkingSpaceId";

                        await connection.ExecuteAsync(updateSql,
                            new { Status = newStatus, ParkingSpaceId = parkingSpaceId },
                            transaction);

                        // Záznam do historie
                        string historySql = @"
                        INSERT INTO StatusHistory (parking_space_id, status, change_time)
                        VALUES (@ParkingSpaceId, @Status, @ChangeTime)";

                        await connection.ExecuteAsync(historySql,
                            new { ParkingSpaceId = parkingSpaceId, Status = newStatus, ChangeTime = DateTime.Now },
                            transaction);

                        transaction.Commit();
                        return true;
                    }
                    catch
                    {
                        transaction.Rollback();
                        throw;
                    }
                }
            }
        }
    }
}
