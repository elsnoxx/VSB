using System;
using System.Collections.Generic;
using System.Data;
using System.Data.SqlClient;
using System.Linq;
using System.Net.Http.Headers;
using System.Runtime.ConstrainedExecution;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;

using StoreIS.orm;
using StoreIS.orm.dto;

namespace StoreIS.orm.dao
{
    public class TransactionsDao
    {
        public static bool InsertProductInOrder(Database pDb, ref int? p_id_order, int p_id_user, int p_id_staff, 
            int p_id_product, int p_quantity)
        {
            Database db = Database.Connect(pDb);

            bool ret = true;
            try
            {
                db.BeginTransaction();

                Order o = new Order();
                o.id_user = p_id_user;
                o.id_staff = p_id_staff;
                o.date_order = DateTime.Today;
                o.price = null;

                if (p_id_order == null) {
                    OrderDao.Insert(db, o);
                }
                else {
                    o.id_order = (int)p_id_order;
                }

                int v_quantity = QueriesDao.ProductQuantity(db, p_id_product);

                if (v_quantity >= p_quantity)
                {
                    OrderItem oi = new OrderItem();
                    oi.id_product = p_id_product;
                    oi.quantity = p_quantity;

                    OrderItemDao.Insert(db, oi, o);
                }

                p_id_order = o.id_order;

                db.EndTransaction();
            }
            catch (SqlException)
            {
                db.Rollback();
                ret = false;
            }

            Database.Close(pDb, db);

            return ret;
        }

        public static bool InsertProductInOrder_sp(Database pDb, ref int? p_id_order, int p_id_user, int p_id_staff,
            int p_id_product, int p_quantity)
        {
            Database db = Database.Connect(pDb);

            bool v_ret = false;

            SqlCommand command = db.CreateCommand("InsertProductInOrder");
            command.CommandType = CommandType.StoredProcedure;

            // Prepare parameters
            SqlParameter p1 = new SqlParameter("@p_id_order", p_id_order == null ? DBNull.Value : (object)p_id_order);
            p1.Direction = ParameterDirection.InputOutput;
            p1.Size = sizeof(int);
            command.Parameters.Add(p1);

            command.Parameters.AddWithValue("@p_id_user", p_id_user);
            command.Parameters.AddWithValue("@p_id_staff", p_id_staff);
            command.Parameters.AddWithValue("@p_id_product", p_id_product);
            command.Parameters.AddWithValue("@p_quantity", p_quantity);

            SqlParameter p6 = new SqlParameter("@p_ret", SqlDbType.Bit);
            p6.Direction = ParameterDirection.Output;
            command.Parameters.Add(p6);

            db.ExecuteNonQuery(command);

            // Process output parameters
            p_id_order = Convert.ToInt32(p1.Value);
            v_ret = (bool)p6.Value;

            Database.Close(pDb, db);

            return v_ret;
        }
    }
}
