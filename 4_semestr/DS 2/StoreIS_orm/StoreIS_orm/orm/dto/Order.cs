using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace StoreIS.orm.dto
{
    public class Order
    {
        public int id_order { get; set; }
        public int id_user { get; set; }
        public int id_staff { get; set; }
        public DateTime date_order { get; set; }
        public float? price { get; set; }
    }
}
