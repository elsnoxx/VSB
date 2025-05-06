using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using StoreIS.orm;
using StoreIS.orm.dao;

namespace StoreIS_orm
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Database db = new Database();
            db.Connect();

            int? id_order = null;
            int id_user = 1;
            int id_staff = 1;
            int id_product = 1;
            int quantity = 2;

            bool ret1 = TransactionsDao.InsertProductInOrder(db, ref id_order, id_user, id_staff, id_product, quantity);
            Console.WriteLine("InsertProductInOrder: ret: " + ret1 + ", id_order: " + id_order);

            id_product = 2;
            bool ret2 = TransactionsDao.InsertProductInOrder_sp(db, ref id_order, id_user, id_staff, id_product, quantity);
            Console.WriteLine("InsertProductInOrder_sp: ret: " + ret2 + ", id_order: " + id_order);

            bool ret3 = TransactionsDao.InsertProductInOrder(db, ref id_order, id_user, id_staff, id_product, quantity);
            Console.WriteLine("InsertProductInOrder: ret: " + ret3 + ", id_order: " + id_order);

            id_order = null;
            id_product = 1;
            bool ret4 = TransactionsDao.InsertProductInOrder_sp(db, ref id_order, id_user, id_staff, id_product, quantity);
            Console.WriteLine("InsertProductInOrder_sp: ret: " + ret4 + ", id_order: " + id_order);

            db.Close();
        }
    }
}
