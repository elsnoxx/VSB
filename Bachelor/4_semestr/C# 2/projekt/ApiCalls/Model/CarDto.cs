using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ApiCalls.Model
{
    public class CarDto
    {
        public int carId { get; set; }
        public int userId { get; set; }
        public string licensePlate { get; set; } = string.Empty;
        public string brandModel { get; set; } = string.Empty;
        public string color { get; set; } = string.Empty;
    }
}
