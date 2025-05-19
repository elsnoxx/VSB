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
            return await conn.QueryFirstOrDefaultAsync<ParkingLotDto>(
                @"SELECT 
                    parking_lot_id AS ParkingLotId,
                    name AS Name,
                    latitude AS Latitude,
                    longitude AS Longitude,
                    capacity AS Capacity
                  FROM ParkingLot
                  WHERE parking_lot_id = @id", new { id });
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
            var sql = "UPDATE ParkingLot SET name = @Name, latitude = @Latitude, longitude = @Longitude WHERE parking_lot_id = @ParkingLotId";
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
                    (pl.capacity - COUNT(ps.parking_space_id)) AS FreeSpaces
                FROM ParkingLot pl
                LEFT JOIN ParkingSpace ps ON pl.parking_lot_id = ps.parking_lot_id AND ps.status = 'occupied'
                GROUP BY pl.parking_lot_id";
            return await conn.QueryAsync<ParkingLot>(sql);
        }
    }
}