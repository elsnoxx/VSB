using StoreIS.orm.dto;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StoreIS.orm.dao
{
    public class QueriesDao
    {
        private static string SqlProductQuantity = 
            "select (" + 
            " coalesce((select sum(quantity) from Supply where id_product = @id_product), 0) -" +
            " coalesce((select sum(quantity) from OrderItem where id_product=@id_product), 0)"+ 
            ");";

        public static int ProductQuantity(Database pDb, int? p_id_product)
        {
            Database db = Database.Connect(pDb);

            int v_quantity = 0;

            SqlCommand command = db.CreateCommand(SqlProductQuantity);
            command.Parameters.AddWithValue("@id_product", p_id_product);
            SqlDataReader reader = db.Select(command);

            while (reader.Read())
            {
                v_quantity = Convert.ToInt32(reader.GetValue(0));
                break;
            }
            reader.Close();

            Database.Close(pDb, db);

            return v_quantity;
        }
    }
}
