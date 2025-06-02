using Dapper;
using ParkingLotWEB.Models;
using System.Data;
using ParkingLotWEB.Models.Entities;

namespace ParkingLotWEB.Database
{
    public class ParkingLotRepository
    {
        private readonly DapperRepository _dapper;

        public ParkingLotRepository(IConfiguration config)
        {
            _dapper = new DapperRepository(config);
        }

        public async Task<IEnumerable<ParkingLot>> GetAllAsync()
        {
            using var conn = _dapper.CreateConnection();
            return await conn.QueryAsync<ParkingLot>("SELECT parking_lot_id AS ParkingLotId, name, latitude, longitude FROM ParkingLot");
        }

        public async Task<ParkingLotDto?> GetByIdAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"
                SELECT 
                    pl.parking_lot_id AS ParkingLotId,
                    pl.name AS Name,
                    pl.latitude AS Latitude,
                    pl.longitude AS Longitude,
                    pl.capacity AS Capacity,
                    (pl.capacity - COUNT(ps.parking_space_id)) AS FreeSpaces,
                    pl.price_per_hour AS PricePerHour
                FROM ParkingLot pl
                LEFT JOIN ParkingSpace ps 
                    ON pl.parking_lot_id = ps.parking_lot_id AND ps.status = 'occupied'
                WHERE pl.parking_lot_id = @id
                GROUP BY 
                    pl.parking_lot_id, 
                    pl.name, 
                    pl.latitude, 
                    pl.longitude, 
                    pl.capacity, 
                    pl.price_per_hour";

            return await conn.QuerySingleOrDefaultAsync<ParkingLotDto>(sql, new { id });
        }

        public async Task<int> CreateAsync(ParkingLot lot)
        {
            using var conn = _dapper.CreateConnection();
            // Vložení parkoviště a získání ID
            var sql = "INSERT INTO ParkingLot (name, latitude, longitude, capacity) VALUES (@Name, @Latitude, @Longitude, @Capacity); SELECT LAST_INSERT_ID();";
            var parkingLotId = await conn.ExecuteScalarAsync<int>(sql, lot);

            // Vytvoření parkovacích míst
            var spaces = new List<object>();
            for (int i = 1; i <= lot.Capacity; i++)
            {
                spaces.Add(new { parking_lot_id = parkingLotId, space_number = i, status = "available" });
            }
            var spaceSql = "INSERT INTO ParkingSpace (parking_lot_id, space_number, status) VALUES (@parking_lot_id, @space_number, @status)";
            await conn.ExecuteAsync(spaceSql, spaces);

            return parkingLotId;
        }

        public async Task<int> UpdateAsync(ParkingLot lot)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE ParkingLot SET name = @Name, latitude = @Latitude, longitude = @Longitude, price_per_hour = @PricePerHour WHERE parking_lot_id = @ParkingLotId";
            return await conn.ExecuteAsync(sql, lot);
        }

        public async Task<int> DeleteAsync(int id)
        {
            using var conn = _dapper.CreateConnection();
            var deleteSpacesSql = "DELETE FROM ParkingSpace WHERE parking_lot_id = @id";
            await conn.ExecuteAsync(deleteSpacesSql, new { id });

            var sql = "DELETE FROM ParkingLot WHERE parking_lot_id = @id";
            return await conn.ExecuteAsync(sql, new { id });
        }

        public async Task<IEnumerable<ParkingLot>> GetAllWithFreeSpacesAsync()
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"
                SELECT 
                    pl.parking_lot_id AS ParkingLotId,
                    pl.name AS Name,
                    pl.latitude AS Latitude,
                    pl.longitude AS Longitude,
                    pl.capacity AS Capacity,
                    (pl.capacity - COUNT(ps.parking_space_id)) AS FreeSpaces,
                    price_per_hour AS PricePerHour
                FROM ParkingLot pl
                LEFT JOIN ParkingSpace ps ON pl.parking_lot_id = ps.parking_lot_id AND ps.status = 'occupied'
                GROUP BY pl.parking_lot_id";
            return await conn.QueryAsync<ParkingLot>(sql);
        }

        public async Task<Dictionary<int, int>> GetCompletedParkingsLastMonthAsync()
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"
                SELECT parking_lot_id, COUNT(*) AS completed_parkings
                FROM ParkingHistory
                WHERE departure_time IS NOT NULL
                AND departure_time >= DATE_SUB(NOW(), INTERVAL 1 MONTH)
                GROUP BY parking_lot_id";
            var result = await conn.QueryAsync<(int parking_lot_id, int completed_parkings)>(sql);
            return result.ToDictionary(x => x.parking_lot_id, x => x.completed_parkings);
        }

        public async Task<IEnumerable<OccupancyPointDto>> GetOccupancyTimelineAsync(int parkingLotId)
        {
            using var conn = _dapper.CreateConnection();
            var sql = @"
                SELECT 
                    sh.change_time AS Time,
                    SUM(CASE WHEN sh.status = 'occupied' THEN 1 ELSE 0 END) AS OccupiedSpaces
                FROM StatusHistory sh
                JOIN ParkingSpace ps ON sh.parking_space_id = ps.parking_space_id
                WHERE ps.parking_lot_id = @parkingLotId
                GROUP BY sh.change_time
                ORDER BY sh.change_time";
            return await conn.QueryAsync<OccupancyPointDto>(sql, new { parkingLotId });
        }

        public async Task<int> UpdatePricePerHourAsync(int parkingLotId, decimal pricePerHour)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "UPDATE ParkingLot SET price_per_hour = @pricePerHour WHERE parking_lot_id = @parkingLotId";
            return await conn.ExecuteAsync(sql, new { pricePerHour, parkingLotId });
        }

        public decimal GetPricePerHourByParkingLotName(string parkingLotName)
        {
            using var conn = _dapper.CreateConnection();
            var sql = "SELECT price_per_hour FROM ParkingLot WHERE name = @parkingLotName";
            return conn.QuerySingleOrDefault<decimal>(sql, new { parkingLotName });
        }
        
    }
}