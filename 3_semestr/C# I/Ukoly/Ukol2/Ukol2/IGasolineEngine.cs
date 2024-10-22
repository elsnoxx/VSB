using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Ukol2
{
    internal interface IGasolineEngine
    {
        double FuelTankSize { init;  }
        double AmountOfFuel { get; }
        void Refuel(double quantity);
    }
}
