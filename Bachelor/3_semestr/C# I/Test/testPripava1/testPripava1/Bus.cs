using System.Text;

namespace testPripava1;

public class Bus : Vehicle, ITicketPrice
{
   public double? TicketPrice { get; set; }

   public Bus(double speed, double? ticketPrice = null) : base(speed)
   {
      this.TicketPrice = ticketPrice;
   }

   public override string ToString()
   {
      StringBuilder sb = new();

      sb.Append("Autobus s maximální rychlostí ");
      sb.Append(base.Speed);
      sb.Append("km/h a cenou lístku ");
      sb.Append(TicketPrice == null ? "???" :  TicketPrice.ToString());
      sb.Append("Kč");
      return sb.ToString();
   }

   public double? GetTicketPrice()
   {
      return TicketPrice;
   }
   
}