using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using StoreIS.orm.dto;

namespace StoreIS.orm.dao
{
    public class OrderDao
    {
        static string SqlInsert = 
            "insert into \"Order\"(id_user, id_staff, date_order, price) " +
            "output Inserted.id_order " +
            "values(@id_user, @id_staff, @date_order, @price)";
        public static void Insert(Database pDb, Order order)
        {
            Database db = Database.Connect(pDb);

            SqlCommand command = db.CreateCommand(SqlInsert);
            PrepareCommand(command, order);
            order.id_order = db.ExecuteScalar(command);

            Database.Close(pDb, db);
        }

        private static void PrepareCommand(SqlCommand command, Order order)
        {
            command.Parameters.AddWithValue("@id_user", order.id_user);
            command.Parameters.AddWithValue("@id_staff", order.id_staff);
            command.Parameters.AddWithValue("@date_order", order.date_order);
            command.Parameters.AddWithValue("@price", order.price == null ? DBNull.Value : (object)order.price);
        }
    }
}
