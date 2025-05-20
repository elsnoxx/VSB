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
                SELECT 
                    parking_space_id AS ParkingSpaceId,
                    parking_lot_id AS ParkingLotId,
                    space_number AS SpaceNumber,
                    status AS Status
                FROM ParkingSpace
                WHERE parking_lot_id = @ParkingLotId 
                AND status = 'available'";
            return await _repository.QueryAsync<ParkingSpace>(sql, new { ParkingLotId = parkingLotId });
        }

        // Aktualizace stavu parkovacího místa a záznam historie
        public async Task<bool> UpdateStatusAsync(int parkingSpaceId, string newStatus)
        {
            using var conn = _repository.CreateConnection();
            var sql = "UPDATE ParkingSpace SET status = @Status WHERE parking_space_id = @Id";
            var affected = await conn.ExecuteAsync(sql, new { Status = newStatus, Id = parkingSpaceId });
            Console.WriteLine($"UPDATE affected rows: {affected}");
            return affected > 0;
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

            // Změň status na 'occupied'
            string updateStatusSql = @"UPDATE ParkingSpace SET status = 'occupied' WHERE parking_space_id = @ParkingSpaceId";
            await _repository.ExecuteAsync(updateStatusSql, new { ParkingSpaceId = parkingSpaceId });
        }

        // Uvolnění obsazenosti parkovacího místa
        public async Task ReleaseOccupancyAsync(int parkingSpaceId, int parkingLotId)
        {
             await _repository.ExecuteInTransactionAsync(async (conn, tran) =>
            {
                // 1. Najdi otevřenou obsazenost
                var occupancy = await conn.QueryFirstOrDefaultAsync<dynamic>(
                    @"SELECT o.occupancy_id, o.license_plate, o.start_time, ps.parking_lot_id, c.car_id
                      FROM Occupancy o
                      JOIN ParkingSpace ps ON o.parking_space_id = ps.parking_space_id
                      JOIN Car c ON o.license_plate = c.license_plate
                      WHERE o.parking_space_id = @ParkingSpaceId AND o.end_time IS NULL
                      ORDER BY o.start_time DESC
                      LIMIT 1",
                    new { ParkingSpaceId = parkingSpaceId }, tran);

                if (occupancy == null)
                    return;

                var endTime = DateTime.Now;

                // 2. Aktualizuj Occupancy
                await conn.ExecuteAsync(
                    @"UPDATE Occupancy
                      SET end_time = @EndTime,
                          duration = TIMESTAMPDIFF(MINUTE, start_time, @EndTime),
                          price = TIMESTAMPDIFF(MINUTE, start_time, @EndTime) * 1.0
                      WHERE occupancy_id = @OccupancyId",
                    new { EndTime = endTime, OccupancyId = occupancy.occupancy_id }, tran);

                // 3. Vlož do ParkingHistory
                await conn.ExecuteAsync(
                    @"INSERT INTO ParkingHistory (car_id, parking_lot_id, arrival_time, departure_time)
                      VALUES (@CarId, @ParkingLotId, @ArrivalTime, @DepartureTime)",
                    new
                    {
                        CarId = occupancy.car_id,
                        ParkingLotId = occupancy.parking_lot_id,
                        ArrivalTime = occupancy.start_time,
                        DepartureTime = endTime
                    }, tran);

                // 4. Změň status na 'available'
                await conn.ExecuteAsync(
                    @"UPDATE ParkingSpace SET status = 'available' WHERE parking_space_id = @ParkingSpaceId",
                    new { ParkingSpaceId = parkingSpaceId }, tran);
            });
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
            var sql = @"
                SELECT 
                    parking_space_id AS ParkingSpaceId,
                    parking_lot_id AS ParkingLotId,
                    space_number AS SpaceNumber,
                    status AS Status
                FROM ParkingSpace
                WHERE parking_lot_id = @ParkingLotId";
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
                            c.user_id AS OwnerId,
                            c.car_id AS CarId
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

        public async Task<IEnumerable<string>> GetAllOccupiedLicensePlatesAsync()
        {
            using var conn = _repository.CreateConnection();
            var sql = @"
                SELECT o.license_plate
                FROM ParkingSpace ps
                JOIN Occupancy o ON ps.parking_space_id = o.parking_space_id
                WHERE o.end_time IS NULL";
            return await conn.QueryAsync<string>(sql);
        }
    }
}
