using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StoreIS.orm.dto
{
    public class OrderItem
    {
        public int id_order { get; set; }
        public int id_product { get; set; }
        public float unit_price { get; set; }
        public float quantity { get; set; }
    }
}
