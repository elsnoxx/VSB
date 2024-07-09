import Data.List
import Data.Text.Lazy.Builder.RealFloat (realFloat)
import Distribution.TestSuite (TestInstance(name))
--Napište funkci, která vrátí symboly na sudých pozicích v řetězci
--evens "ABCDEF" = "ACE"
evens :: String -> String
evens [] = []
evens (x:y:z) = x : evens z



--Napište funkci, která v libovolném řetězci zamění poslední znak za ‘!’
--swaap "AHOJ" = AHO!
swaap :: String -> String
swaap [] = []
swaap [x] = ['!']
swaap (x:xs) =  x : swaap xs



--Napište funkci, která bude mít na vstupu mužské příjmení a doplní k němu přechýlení na ženské příjmení
--makeGirl "Novak" = "Novakova"
makeGirl :: String -> String
makeGirl x = x ++ "ova"


--Napište funkci, která v libovolném řetězci nahradí výskyt jednoho znaku za jiný znak.
--Oba znaky jsou parametry funkce
--swap "ABCDABCD" 'A' 'X' = "XBCDXBCD"
swap :: String -> Char -> Char -> String
swap [] _ _ = []
swap (x:xs) a b = if x == a then b : swap xs a b else x : swap xs a b


--Napište funkci, která vrátí symboly na lichých pozicích v řetězci
--odds "ABCDEF" = "BDF"
odds :: String -> String
odds [] = []
odds (x:y:z) = y : odds z



--Napište funkci, která spočítá počet pravých a levých závorek ve vstupní výrazu datového typu String
--Funkce vrátí True, pokud je jejich počet shodný, jinak False
-- check "((()))()" = True 
-- check "(()" = False

check :: String -> Bool
pocet retezec pismeno = length $ filter (==pismeno) retezec
check text = (pocet text '(') == (pocet text ')')



--Napište funkci, která vrátí dvojici složenou s nejmenšího a největšího čísla v seznamu hodnot typu Num
--interval [2,3,4,5,1,2,3,5,7,4] = (1,7)
interval :: [Int] -> (Int,Int)
interval i = (a,b) 
            where 
                a = minimum i 
                b = maximum i

--Vytvořte seznam který bude obsahovat prvních n mocnin čísla x
--Číslo x a číslo n jsou parametry vstupu
--powers 2 6 = [1,2,4,8,16,32] 
--powers 10 3 = [1,10,100]
powers :: Int -> Int -> [Int]
powers a b = [a^b | b <- [1..b]] 



--Mějme seznam čísel typu Double, spočítejte průměr z těchto čísel
--average [1,2,3,4] = 2.5
average :: [Float] -> Float
average [] = 0
average xs = realToFrac (sum xs) / genericLength xs

--Mějme tři seznamy s prvky stejného typu, napište funkci, která najde a vrátí prvky, které jsou ve všech třech seznamech
--intersection3 "ABCD" "XXAB" "XAX" = "A"
-- intersection3 :: Eq a => [a] -> [a] -> [a] -> [a]
-- intersection3 xs ys zs = [y| y<-ys, elem y xs, elem y zs]   


--Mějme na vstupu seznam dvojic (osoba, vek), vraťte seznam osob, jejichž věk je menší než číslo a na vstupu
--filter' [("James",40),("Penny",20)] 30 = ["Penny"]
filter' :: [(String,Int)] -> Int -> [String]
filter' [] _ = []   
filter' (x:xs) n = if snd x < n then [fst x] ++ filter' xs n else filter' xs n



--Mějme seznam celých čísel, najděte číslo, které se opakuje nejvíce krát
--Je-li jich více, stačí jedno z nich
--mostFrequent [1,2,3,1,2,3,1] = 1

mostFrequent :: [Int] -> (Int,Int)
mostFrequent ns = maximum [ (length ks, head ks) | ks <- group (sort ns) ]


--Mějme na vstupu seznam dvojic (Jmeno, Prijmeni) a řetězec vzor, vraťte jména všech, jejichž příjmení obsahuje zadaný vzor
--names [("Marek","Behalek"), ("Martin","Kot"), ("Michal","Vasinek")] "ek" = ["Marek","Michal"].
names ::  [(String,String)] -> String -> [String]
names [] _ = []
names (x:xs) zs 
    | isSubstring zs (snd x) == True = (fst x) : names xs zs
    | otherwise = names xs zs