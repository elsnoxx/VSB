using System;
using System.Data;
using System.Threading.Tasks;
using Dapper;
using Microsoft.Extensions.Configuration;
using MySql.Data.MySqlClient;

public class DapperRepository
{
    // privet or 
    public readonly string _connectionString;

    public DapperRepository(IConfiguration configuration)
    {
        _connectionString = configuration.GetConnectionString("DefaultConnection")
            ?? throw new ArgumentNullException("Connection string cannot be null.");
    }

    // Pro manuální předání connection stringu
    public DapperRepository(string connectionString)
    {
        _connectionString = connectionString;
    }

    // Vytvoření nového připojení
    private IDbConnection CreateConnection()
    {
        return new MySqlConnection(_connectionString);
    }

    public async Task<IEnumerable<T>> QueryAsync<T>(string sql, object parameters = null)
    {
        try
        {
            using (var connection = CreateConnection())
            {
                if (connection is MySqlConnection mySqlConnection)
                {
                    await mySqlConnection.OpenAsync();
                }
                return await connection.QueryAsync<T>(sql, parameters);
            }
        }
        catch (Exception ex)
        {
            throw new Exception("An error occurred while querying the database.", ex);
        }
    }

    public async Task<int> ExecuteAsync(string sql, object parameters = null)
    {
        using (var connection = CreateConnection())
        {
            if (connection is MySqlConnection mySqlConnection)
            {
                await mySqlConnection.OpenAsync();
            }
            return await connection.ExecuteAsync(sql, parameters);
        }
    }

}