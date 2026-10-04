using StoreIS.orm.dto;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StoreIS.orm.dao
{
    public class OrderItemDao
    {
        static string SqlInsert =
          "insert into OrderItem(id_order, id_product, unit_price, quantity) " +
          "values(@p_id_order, @p_id_product, " +
          "  (select p.unit_price* " +
          "    (select " +
          "       case u.level " +
          "         when 2 then 0.95 " +
          "         when 3 then 0.9 " +
          "         when 4 then 0.85 " +
          "         when 5 then 0.8 " +
          "         when 6 then 0.75 " +
          "         when 7 then 0.7 " +
          "         else 1 " +
          "       end " +
          "     from \"User\" u where id_user = @p_id_user " +
          "    ) " +
          "   from Product p " +
          "   where p.id_product = @p_id_product " +
          "  ), @p_quantity " +
          ");";

        public static void Insert(Database pDb, OrderItem oi, Order o)
        {
            Database db = Database.Connect(pDb);

            SqlCommand command = db.CreateCommand(SqlInsert);
            PrepareCommand(command, oi, o);
            db.ExecuteNonQuery(command);

            Database.Close(pDb, db);
        }

        private static void PrepareCommand(SqlCommand command, OrderItem oi, Order o)
        {
            command.Parameters.AddWithValue("@p_id_order", o.id_order);
            command.Parameters.AddWithValue("@p_id_product", oi.id_product);
            command.Parameters.AddWithValue("@p_id_user", o.id_user);
            command.Parameters.AddWithValue("@p_quantity", oi.quantity);
        }
    }
}
