--import Data.Char
type Result = [String]
pp :: Result -> IO ()
pp x = putStr (concat (map (++"\n") x))


smer 'u' = (0,-1)
smer 'd' = (0,1)
smer 'l' = (-1,0)
smer 'r' = (1,0)


myElem :: (Eq a) => a -> [a] -> Bool
myElem _ [] = False 
myElem x (y:ys)
    | x == y    = True 
    | otherwise = myElem x ys 

first' :: (a, b) -> a
first' (x, _) = x


second :: (a, b) -> b
second (_, y) = y


allPos [] _ = []
allPos ((move, delka):xs) (sci,sri) = let
    (offc, offR) = smer move
    celek = [(sci + x*offc , sri + offR*x)   
            |x<-[0..delka] ]
    in celek ++ allPos xs (last celek)

draw :: [(Char, Int)] -> Result

draw moves = let
    moves' = allPos moves (0,0)
    (minCol, maxCol) = (minimum (map first' moves'), maximum (map first' moves'))
    (minRow, maxRow) = (minimum (map second moves'), maximum (map second moves'))
    in [[if (myElem (col,row) moves') then '#' else ' '
    | col<-[minCol..maxCol]] 
    | row<-[minRow..maxRow]]




