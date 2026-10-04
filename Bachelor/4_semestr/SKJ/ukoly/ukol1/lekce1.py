def count_words(data):
    """
    Return a dictionary that maps word -> number of occurences in `data`.
    Words are separated by spaces (' ').
    Characters are case sensitive.

    Hint:
        "hi there".split(" ") -> ["hi", "there"]

    Example:
        count_words('this car is my favourite what car is this')
        {
            'this': 2,
            'car': 2,
            'is': 2,
            'my': 1,
            'favourite': 1,
            'what': 1
        }
    """



assert count_words('hello is this the crusty crab no this is patrick') == \
{
    'hello': 1,
    'this': 2,
    'is': 2,
    'the': 1,
    'crusty': 1,
    'crab': 1,
    'no': 1,
    'patrick': 1
}
assert count_words('what happens in kernel mode stays in kernel mode') == \
{
    'what': 1,
    'happens': 1,
    'in': 2,
    'kernel': 2,
    'mode': 2,
    'stays': 1
}
assert count_words('') == {}