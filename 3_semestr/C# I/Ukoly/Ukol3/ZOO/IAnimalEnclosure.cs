using ZOO.Animals;

namespace ZOO
{
    public interface IAnimalEnclosure
    {
        string Name { get; }
        void MakeAnimalsSound();
        List<T> GetAnimals<T>() where T : class;
    }
}
