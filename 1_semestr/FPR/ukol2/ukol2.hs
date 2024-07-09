import Data.List

data Point = Point Int Int



data Shape
  = Ellipse Point Int
  | Square {topLeft :: Point, size :: Point}

type Result = [String]

pp :: Result -> IO ()
pp x = putStr (concat (map (++ "\n") x))

matrix :: (Int, Int) -> [((Int, Int), Char)]
matrix (cols, rows) = concat [[((x, y), '.') | x <- [0 .. cols -1]] | y <- [0 .. rows -1]]

sortIt :: [((Int, Int), Char)] -> [((Int, Int), Char)]
sortIt list = tmp list
  where
    tmp [] = []
    tmp (head : rest) = sortIt leftSide ++ [head] ++ sortIt rightSide
      where
        ((x0, y0), _) = head
        leftSide = [((x, y), c) | ((x, y), c) <- rest, y * cols + x < y0 * cols + x0]
        rightSide = [((x, y), c) | ((x, y), c) <- rest, y * cols + x >= y0 * cols + x0]
        cols = maximum [snd a | (a, _) <- list] + 1

toResult :: [((Int, Int), Char)] -> Result
toResult matrix = [[b | (a, b) <- sorted, snd a == row] | row <- [0 .. rows -1]]
  where
    sorted = sortIt matrix
    rows = maximum [snd a | (a, _) <- sorted] + 1

draw :: [((Int, Int), Char)] -> [Shape] -> [((Int, Int), Char)]
draw matrix [] = matrix
draw matrix (x : xs) = draw (drawShape matrix x) xs

drawShape :: [((Int, Int), Char)] -> Shape -> [((Int, Int), Char)]
drawShape matrix (Square start end) = drawBox matrix start end
drawShape matrix (Ellipse center radius) = drawEllipse matrix center radius

drawBox :: [((Int, Int), Char)] -> Point -> Point -> [((Int, Int), Char)]
drawBox matrix a b = (matrix \\ rest) ++ modifiedRest
  where
    rest = [((x, y), c) | ((x, y), c) <- matrix, x >= x1 && x <= x2 && y >= y1 && y <= y2 && (x == x1 || x == x2 || y == y1 || y == y2)]
    modifiedRest = [((x, y), '#') | ((x, y), _) <- matrix, x >= x1 && x <= x2 && y >= y1 && y <= y2 && (x == x1 || x == x2 || y == y1 || y == y2)]
    Point x1 y1 = a
    Point x2 y2 = b

drawEllipse :: [((Int, Int), Char)] -> Point -> Int -> [((Int, Int), Char)]
drawEllipse matrix s r = (matrix \\ rest) ++ modifiedRest
  where
    rest = [((x, y), c) | ((x, y), c) <- matrix, (x - x1) ^ 2 + (y - y1) ^ 2 >= r ^ 2 - range && (x - x1) ^ 2 + (y - y1) ^ 2 <= r ^ 2 + range]
    modifiedRest = [((x, y), '#') | ((x, y), _) <- matrix, (x - x1) ^ 2 + (y - y1) ^ 2 >= r ^ 2 - range && (x - x1) ^ 2 + (y - y1) ^ 2 <= r ^ 2 + range]
    Point x1 y1 = s
    range = r `div` 2


view :: (Int, Int) -> [Shape] -> Result
view (cols, rows) shapes = let empty = matrix (cols, rows) in toResult (draw empty shapes)