using System.Text;

namespace testPripava1;

public class Truck : Vehicle, IGoodsTransportPrice
{
    public double GoodsTransportPrice { get; set; }
    
    public Truck(double speed, double goodsTransportPrice) : base(speed)
    {
        this.GoodsTransportPrice = goodsTransportPrice;
    }

    public override string ToString()
    {
        StringBuilder sb = new();

        sb.Append("Nákladní auto s maximální rychlostí ");
        sb.Append(base.Speed);
        sb.Append("km/h a cenou přepravy zboží ");
        sb.Append(GoodsTransportPrice);
        sb.Append("Kč");
        return sb.ToString();
    }

    public double GetGoodsTransportPrice()
    {
        return GoodsTransportPrice;
    }
}