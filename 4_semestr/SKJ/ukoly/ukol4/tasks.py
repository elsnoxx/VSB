import time


class Vector:
    """
    Implement the methods below to create a 3D vector class.

    Magic methods cheatsheet: https://rszalski.github.io/magicmethods
    """

    """
    Implement a constructor that takes three coordinates (x, y, z) and stores
    them as attributes with the same names in the Vector.
    Default value for all coordinates should be 0.
    Example:
        v = Vector(1.2, 3.5, 4.1)
        v.x # 1.2
        v = Vector(z=1) # == Vector(0, 0, 1)
    """
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z

    def __setitem__(self, key, value):
        if key == 0:
            self.x = value
        elif key == 1:
            self.y = value
        elif key == 2:
            self.z = value
        else:
            raise IndexError("Index out of range. Valid indices are 0, 1, and 2.")
            

    """
    Implement vector addition and subtraction using `+` and `-` operators.
    Both operators should return a new vector and not modify its operands.
    If the second operand isn't a vector, raise ValueError.
    Example:
        Vector(1, 2, 3) + Vector(4, 5, 6) # Vector(5, 7, 8)
        Vector(1, 2, 3) - Vector(4, 5, 6) # Vector(-3, -3, -3)
    Hint:
        You can use isinstance(object, class) to check whether `object` is an instance of `class`.
    """
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y, self.z + other.z)
        else:
            raise ValueError
        
    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y, self.z - other.z)
        else:
            raise ValueError

    """
    Implement the `==` comparison operator for Vector that returns True if both vectors have the same attributes.
    If the second operand isn't a vector, return False.
    Example:
        Vector(1, 1, 1) == Vector(1, 1, 1)  # True
        Vector(1, 1, 1) == Vector(2, 1, 1)  # False
        Vector(1, 2, 3) == 5                # False
    """
    def __eq__(self, other):
        if isinstance(other, Vector):
            if self.x == other.x and self.y == other.y and self.z == other.z:
                return True
            else:
                return False
        else:
            return False

    """
    Implement string representation of Vector in the form `(x, y, z)`.
    Example:
        str(Vector(1, 2, 3))    # (1, 2, 3)
        print(Vector(0, 0, 0))  # (0, 0, 0)
    """
    def __str__(self):
        text = f"({self.x}, {self.y}, {self.z})"
        return text

    """
    Implement indexing for the vector, both for reading and writing.
    If the index is out of range (> 2), raise IndexError.
    Example:
        v = Vector(1, 2, 3)
        v[0] # 1
        v[2] # 3
        v[1] = 5 # v.y == 5

        v[10] # raises IndexError
    """
    def __getitem__(self, key):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        elif key == 2:
            return self.z
        else:
            raise IndexError

    """
    Implement the iterator protocol for the vector.
    Hint:
        Use `yield`.
    Example:
        v = Vector(1, 2, 3)
        for x in v:
            print(x) # prints 1, 2, 3
    """
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z 

    
class Observable:
    """
    Implement the `observer` design pattern.
    Observable should have a `subscribe` method for adding new subscribers.
    It should also have a `notify` method that calls all of the stored subscribers and passes them its parameters.
    Example:
        obs = Observable()

        def fn1(x):
            print("fn1: {}".format(x))

        def fn2(x):
            print("fn2: {}".format(x))

        unsub1 = obs.subscribe(fn1)     # fn1 will be called everytime obs is notified
        unsub2 = obs.subscribe(fn2)     # fn2 will be called everytime obs is notified
        obs.notify(5)                   # should call fn1(5) and fn2(5)
        unsub1()                        # fn1 is no longer subscribed
        obs.notify(6)                   # should call fn2(6)
    """
    def __init__(self):
        self.subscribers = []

    def subscribe(self, subscriber):
        """
        Add subscriber to collection of subscribers.
        Return a function that will remove this subscriber from the collection when called.
        """
        self.subscribers.append(subscriber)
        
        def unsubscribe():
            if subscriber in self.subscribers:
                self.subscribers.remove(subscriber)

        return unsubscribe

    def notify(self, *args, **kwargs):
        """
        Pass all parameters given to this function to all stored subscribers by calling them.
        """
        for subscriber in self.subscribers:
            subscriber(*args, **kwargs)


class UpperCaseDecorator:
    """
    Implement the `decorator` design pattern.
    UpperCaseDecorator should decorate a file which will be passed to its constructor.
    It should make all lower case characters written to the file uppercase and remove all
    upper case characters.
    It is enough to support the `write` and `writelines` methods of file.
    Example:
        with open("file.txt", "w") as f:
            decorated = UpperCaseDecorator(f)
            decorated.write("Hello World\n")
            decorated.writelines(["Nice to MEET\n", "YOU"])

        file.txt content after the above code is executed:
        ELLO ORLD
        ICE TO

    """
    def __init__(self, file):
        self.file = file

    def write(self, text):
        new_text = []
        
        for char in text:
            if char.islower():
                new_text.append(char.upper())
            elif char in " \n":
                new_text.append(char)
        
        self.file.write("".join(new_text))

    def writelines(self, lines):
        for line in lines:
            self.write(line)


class GameOfLife:
    """
    Implement "Game of life" (https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

    The game board will be represented with nested tuples, where '.'
    marks a dead cell and 'x' marks a live cell. Cells that are out of bounds of the board are
    assumed to be dead. The board grid will always be a square.

    Try some patterns from wikipedia + the provided tests to test the functionality.

    The GameOfLife objects should be immutable, i.e. the move method will return a new instance
    of GameOfLife.

    Example:
        game = GameOfLife((
            ('.', '.', '.'),
            ('.', 'x', '.'),
            ('.', 'x', '.'),
            ('.', 'x', '.'),
            ('.', '.', '.')
        ))
        game.alive()    # 3
        game.dead()     # 12
        x = game.move() # 'game' doesn't change
        # x.board:
        (
            ('.', '.', '.'),
            ('.', '.', '.'),
            ('x', 'x', 'x'),
            ('.', '.', '.'),
            ('.', '.', '.')
        )

        str(x)
        ...\n
        ...\n
        xxx\n
        ...\n
        ...\n
    """

    def __init__(self, board):
        """
        Create a constructor that receives the game board and stores it in an attribute called
        'board'.
        """
        self.board = board
        pass

    def move(self):
        """
        Simulate one iteration of the game and return a new instance of GameOfLife containing
        the new board state.
        """
        
        newBoard = []
        for i in range(len(self.board)):
            newRow = []
            for j in range(len(self.board[i])):
                alive = 0
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if x == 0 and y == 0:
                            continue
                        if 0 <= i + x < len(self.board) and 0 <= j + y < len(self.board[i]):
                            if self.board[i + x][j + y] == "x":
                                alive += 1
                if self.board[i][j] == "x":
                    if alive < 2 or alive > 3:
                        newRow.append(".")
                    else:
                        newRow.append("x")
                else:
                    if alive == 3:
                        newRow.append("x")
                    else:
                        newRow.append(".")
            newBoard.append(newRow)
            
        newBoardTuple = []
        for row in newBoard:
            newBoardTuple.append(tuple(row))
        
        boardState = tuple(newBoardTuple)
        
        return GameOfLife(boardState)

    def count_cells(self, char):
        cnt = 0
        for row in self.board:
            for cell in row:
                if cell == char:
                    cnt += 1
        return cnt

    def alive(self):
        """
        Return the number of cells that are alive.
        """
        if not self.board:
            raise ValueError("The board is empty. Cannot count alive cells.")
        alive = self.count_cells("x")
        return alive

    def dead(self):
        """
        Return the number of cells that are dead.
        """
        if not self.board:
            raise ValueError("The board is empty. Cannot count dead cells.")
        dead = self.count_cells(".")
        return dead

    def __repr__(self):
        """
        Return a string that represents the state of the board in a single string (with newlines
        for each board row).
        """
        if not self.board:
            return "The board is empty."
        text = ""
        for row in self.board:
            for cell in row:
                text += cell
            text += "\n"
        print(text)
        return text


def play_game(game, n):
    """
    You can use this function to render the game for n iterations
    """
    for i in range(n):
        print(game)
        game = game.move()
        time.sleep(0.25)  # sleep to see the output


# this code will only be executed if you run `python tasks.py`
# it will not be executed when tasks.py is imported
if __name__ == "__main__":
    play_game(GameOfLife((
        ('.', '.', '.'),
        ('.', 'x', '.'),
        ('.', 'x', '.'),
        ('.', 'x', '.'),
        ('.', '.', '.'),
    )), 10)
