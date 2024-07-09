checkBalance :: String -> Bool
checkBalance expression = checkBalance' expression []

checkBalance' :: String -> [Char] -> Bool
checkBalance' [] stack = null stack
checkBalance' (x:xs) stack
    | x == '(' = checkBalance' xs (x:stack)
    | x == ')' = case stack of
        []     -> False
        (y:ys) -> if y == '(' then checkBalance' xs ys else False
    | otherwise = checkBalance' xs stack

main :: IO ()
main = do
    print $ checkBalance "((()))()"  -- True
    print $ checkBalance "(()"       -- False
