using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal interface IElectricEngine
    {
        double BatteryCapacity { get; }
        double RemainingEnergy { get; }
        void Charge(double quantity);
    }
}
