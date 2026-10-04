using System.Text;

namespace testPripava1;

public class Train : Vehicle, ITicketPrice, IGoodsTransportPrice
{
    public double? TicketPrice { get; set; }
    public double GoodsTransportPrice { get; set; }

    public Train(double speed, double? ticketPrice, double goodsTransportPrice) : base(speed)
    {
        this.TicketPrice = ticketPrice;
        this.GoodsTransportPrice = goodsTransportPrice;
    }

    public override string ToString()
    {
        StringBuilder sb = new();

        sb.Append("Vlak s maximální rychlostí ");
        sb.Append(base.Speed);
        sb.Append("km/h, cenou lístku ");
        sb.Append(TicketPrice == null ? "???" : TicketPrice.ToString());
        sb.Append("Kč a cenou přepravy zboží ");
        sb.Append(GoodsTransportPrice);
        sb.Append("Kč");
        return sb.ToString();
    }

    public double? GetTicketPrice()
    {
        return TicketPrice;
    }

    public double GetGoodsTransportPrice()
    {
        return GoodsTransportPrice;
    }
}