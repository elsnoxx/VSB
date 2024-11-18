namespace StockLib.Product;

public interface IProduct : IPhysicalProduct
{
    public string Name { get; }
    public int Price { get; }
}