
-- Mějme na vstupu seznam dvojc (osoba, věk) a číslo n. Vraťte seznam osob, jejichž věk je menší než číslo n na vstupu
-- filter' [("James",40),("Penny",20)] 30
filter' :: [(String,Int)] -> Int -> [String]
filter' [] _ = []   
filter' (x:xs) n = if snd x < n then [fst x] ++ filter' xs n else filter' xs n



-- Napi+ste funkci, které vrátí seznam pozic výskutu písmene v textovem řetezci. Řetězec i písmeno jsou parametry funkcu, Počítání pozic v řetězci začínáme od 0.(*)
-- positions "programovani" 'r' = [1,4]
positions :: String -> Char -> [Int]
positions str ch = [ y | (x, y) <- zip str [0..], x == ch ]





-- Napište funkci, ktera dostane jako parametr tri seznamy a jako vysledek vrati seznam prvku, ktere jsou prave v jednom ze zadanych seznamu.
-- unique [1,2,3,4] [3,4,5,6] [1,6,7]
unique :: Eq a=>[a] -> [a] -> [a] -> [a]
unique xs ys zs = filter (\x -> countInList x xs + countInList x ys + countInList x zs == 1) (xs ++ ys ++ zs)
  where
    countInList :: Eq a => a -> [a] -> Int
    countInList x = length . filter (== x)