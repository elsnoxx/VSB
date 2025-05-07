using Dapper;
using MySql.Data.MySqlClient;
using ParkingLotWEB.Models;
using ParkingLotWEB.Models.Entities;


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
            await _repository.ExecuteInTransactionAsync(async (conn, tran) =>
            {
                string updateSql = @"
                    UPDATE ParkingSpace 
                    SET status = @Status 
                    WHERE parking_space_id = @ParkingSpaceId";
                await conn.ExecuteAsync(updateSql, new { Status = newStatus, ParkingSpaceId = parkingSpaceId }, tran);

                string historySql = @"
                    INSERT INTO StatusHistory (parking_space_id, status, change_time)
                    VALUES (@ParkingSpaceId, @Status, @ChangeTime)";
                await conn.ExecuteAsync(historySql, new { ParkingSpaceId = parkingSpaceId, Status = newStatus, ChangeTime = DateTime.Now }, tran);
            });
            return true;
        }

        // Získání historie stavu parkovacího místa
        public async Task<IEnumerable<StatusHistory>> GetStatusHistoryAsync(int parkingSpaceId)
        {
            string sql = @"SELECT * FROM StatusHistory WHERE parking_space_id = @ParkingSpaceId ORDER BY change_time DESC";
            return await _repository.QueryAsync<StatusHistory>(sql, new { ParkingSpaceId = parkingSpaceId });
        }

        // Vložení obsazenosti parkovacího místa
        public async Task InsertOccupancyAsync(int parkingSpaceId, string licensePlate)
        {
            string sql = @"INSERT INTO Occupancy (parking_space_id, license_plate, start_time) VALUES (@ParkingSpaceId, @LicensePlate, @StartTime)";
            await _repository.ExecuteAsync(sql, new { ParkingSpaceId = parkingSpaceId, LicensePlate = licensePlate, StartTime = DateTime.Now });
        }

        // Uvolnění obsazenosti parkovacího místa
        public async Task ReleaseOccupancyAsync(int parkingSpaceId)
        {
            // Najde poslední otevřenou obsazenost a ukončí ji
            string sql = @"
                UPDATE Occupancy
                SET end_time = @EndTime,
                    duration = TIMESTAMPDIFF(MINUTE, start_time, @EndTime),
                    price = TIMESTAMPDIFF(MINUTE, start_time, @EndTime) * 1.0 -- cena za minutu, upravte dle potřeby
                WHERE parking_space_id = @ParkingSpaceId AND end_time IS NULL
                ORDER BY start_time DESC
                LIMIT 1";
            await _repository.ExecuteAsync(sql, new { ParkingSpaceId = parkingSpaceId, EndTime = DateTime.Now });
        }

        // Získání aktuální obsazenosti parkovacího místa
        public async Task<Occupancy?> GetCurrentOccupancyAsync(int parkingSpaceId)
        {
            string sql = @"SELECT * FROM Occupancy WHERE parking_space_id = @ParkingSpaceId AND end_time IS NULL ORDER BY start_time DESC LIMIT 1";
            var result = await _repository.QueryAsync<Occupancy>(sql, new { ParkingSpaceId = parkingSpaceId });
            return result.FirstOrDefault();
        }

        public async Task<IEnumerable<ParkingSpace>> GetSpacesAsync(int parkingLotId)
        {
            using var conn = _repository.CreateConnection();
            var sql = @"SELECT * FROM ParkingSpace WHERE parking_lot_id = @ParkingLotId";
            return await conn.QueryAsync<ParkingSpace>(sql, new { ParkingLotId = parkingLotId });
        }
        
        public async Task<IEnumerable<ParkingSpaceWithOwner>> GetSpacesWithOwnerAsync(int parkingLotId)
        {
            using var conn = _repository.CreateConnection();
            var sql = @"SELECT 
                            ps.parking_space_id AS ParkingSpaceId,
                            ps.parking_lot_id AS ParkingLotId,
                            ps.space_number AS SpaceNumber,
                            ps.status AS Status,
                            c.user_id AS OwnerId
                        FROM ParkingSpace ps
                        LEFT JOIN Occupancy o ON ps.parking_space_id = o.parking_space_id AND o.end_time IS NULL
                        LEFT JOIN Car c ON o.license_plate = c.license_plate
                        WHERE ps.parking_lot_id = @parkingLotId";
            return await conn.QueryAsync<ParkingSpaceWithOwner>(sql, new { parkingLotId });
        }

        public async Task<IEnumerable<ParkingSpaceWithOwner>> GetSpacesWithDetailsAsync(int parkingLotId)
        {
            using var conn = _repository.CreateConnection();
            var sql = @"SELECT 
                            ps.parking_space_id AS ParkingSpaceId,
                            ps.parking_lot_id AS ParkingLotId,
                            ps.space_number AS SpaceNumber,
                            ps.status AS Status,
                            o.user_id AS OwnerId
                        FROM ParkingSpace ps
                        LEFT JOIN Occupancy o ON ps.parking_space_id = o.parking_space_id AND o.end_time IS NULL
                        WHERE ps.parking_lot_id = @parkingLotId";
            return await conn.QueryAsync<ParkingSpaceWithOwner>(sql, new { parkingLotId });
        }

        public async Task<ParkingSpace?> GetByIdAsync(int parkingSpaceId)
        {
            using var conn = _repository.CreateConnection();
            var sql = @"SELECT * FROM ParkingSpace WHERE parking_space_id = @ParkingSpaceId";
            return await conn.QueryFirstOrDefaultAsync<ParkingSpace>(sql, new { ParkingSpaceId = parkingSpaceId });
        }

        public async Task<ParkingSpaceDetails?> GetParkingSpaceDetailsAsync(int parkingSpaceId)
        {
            using var conn = _repository.CreateConnection();
            var sql = @"SELECT 
                            ps.parking_space_id AS ParkingSpaceId,
                            ps.parking_lot_id AS ParkingLotId,
                            ps.space_number AS SpaceNumber,
                            ps.status AS Status,
                            o.license_plate AS LicensePlate,
                            o.start_time AS StartTime,
                            o.end_time AS EndTime
                        FROM ParkingSpace ps
                        LEFT JOIN Occupancy o ON ps.parking_space_id = o.parking_space_id AND o.end_time IS NULL
                        WHERE ps.parking_space_id = @ParkingSpaceId";
            return await conn.QueryFirstOrDefaultAsync<ParkingSpaceDetails>(sql, new { ParkingSpaceId = parkingSpaceId });
        }
    }
}
