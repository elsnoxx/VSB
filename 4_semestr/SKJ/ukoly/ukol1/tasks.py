def fizzbuzz(num):
    """
    Return 'Fizz' if `num` is divisible by 3, 'Buzz' if `num` is divisible by 5, 'FizzBuzz' if `num` is divisible both by 3 and 5.
    If `num` isn't divisible neither by 3 nor by 5, return `num`.
    Example:
        fizzbuzz(3) # Fizz
        fizzbuzz(5) # Buzz
        fizzbuzz(15) # FizzBuzz
        fizzbuzz(8) # 8
    """
    try:
        if num % 3 == 0 and num % 5 == 0:
            return 'FizzBuzz'
        elif num % 3 == 0:
            return 'Fizz'
        elif num % 5 == 0:
            return 'Buzz'
        
        return num
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}
    pass


def fibonacci(n):
    """
    Return the `n`-th Fibonacci number (counting from 0).
    Example:
        fibonacci(0) == 0
        fibonacci(1) == 1
        fibonacci(2) == 1
        fibonacci(3) == 2
        fibonacci(4) == 3
    """
    if n < 0:
        print("Incorrect input")

    elif n == 0:
        return 0

    elif n == 1 or n == 2:
        return 1

    else:
        return fibonacci(n-1) + fibonacci(n-2)
    pass


def dot_product(a, b):
    """
    Calculate the dot product of `a` and `b`.
    Assume that `a` and `b` have same length.
    Hint:
        lookup `zip` function
    Example:
        dot_product([1, 2, 3], [0, 3, 4]) == 1*0 + 2*3 + 3*4 == 18
    """
    try:
        sum = 0
        products_zip = zip(a,b)
        for product in products_zip:
            sum += product[0] * product[1]

        return sum
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}
    pass


def redact(data, chars):
    """
    Return `data` with all characters from `chars` replaced by the character 'x'.
    Characters are case sensitive.
    Example:
        redact("Hello world!", "lo")        # Hexxx wxrxd!
        redact("Secret message", "mse")     # Sxcrxt xxxxagx
    """
    try:
        data_list = list(data)
        for char in chars:
            for i in range(0,len(data)):
                if char == data_list[i]:
                    data_list[i] = "x"

        return "".join(data_list)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}
    pass


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
    try:
        if not data:
            return {}
        
        splited_data = data.split(' ')

        dict_data = {}
        for i in splited_data:
            if i in dict_data:
                dict_data[i] += 1
            else:
                dict_data[i] = 1
                
        return dict_data
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {}
    pass


def bonus_fizzbuzz(num):
    """
    Implement the `fizzbuzz` function.
    `if`, match-case and cycles are not allowed.
    """
    return (num%15==0 and 'FizzBuzz' or num%3==0 and 'Fizz' or num%5==0 and 'Buzz' or num)
    pass



def bonus_utf8(cp):
    """
    Encode `cp` (a Unicode code point) into 1-4 UTF-8 bytes - you should know this from `Základy číslicových systémů (ZDS)`.
    Example:
        bonus_utf8(0x01) == [0x01]
        bonus_utf8(0x1F601) == [0xF0, 0x9F, 0x98, 0x81]
    """
    if cp < 0 or cp > 0x10FFFF:
        raise ValueError("Invalid Unicode code point")
    
    # Určení počtu bajtů podle velikosti kódového bodu
    if cp < 0x80:
        num_bytes = 1
    elif cp < 0x800:
        num_bytes = 2
    elif cp < 0x10000:
        num_bytes = 3
    else:
        num_bytes = 4

    # Maska pro první bajt podle počtu bajtů
    first_byte_mask = [0b00000000, 0b11000000, 0b11100000, 0b11110000]

    # Inicializace seznamu bajtů
    utf8_bytes = [0] * num_bytes  

    # Naplnění posledních bajtů (každý začíná "10xxxxxx")
    for i in range(num_bytes - 1, 0, -1):
        utf8_bytes[i] = 0b10000000 | (cp & 0b111111)
        cp >>= 6  

    # První bajt s odpovídající maskou
    utf8_bytes[0] = first_byte_mask[num_bytes - 1] | cp

    return utf8_bytes
    
    pass
