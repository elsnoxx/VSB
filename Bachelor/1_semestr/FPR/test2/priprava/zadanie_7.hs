import Data.Text (Text)
import GHC.Float (int2Double)
import Distribution.Simple.Utils (xargs)
import Data.Time.Format.ISO8601 (yearFormat)
import Control.Monad.RWS.Class (MonadState(get))
import Control.Exception (mask)
import Data.Char
import Data.Array (Ix(index))
import Data.ByteString.Builder (FloatFormat)



data TernaryTree a = Leaf a | Branch (TernaryTree a) (TernaryTree a) (TernaryTree a) 

strom = Branch (Leaf 11) (Branch (Leaf 3) (Leaf 8) (Leaf 1)) (Leaf 3)


data Component = TextBox {name :: String, text :: String}
               | Button {name :: String, value :: String}
               | Container {name :: String, children :: [Component]} deriving(Show)

gui :: Component
gui = Container "My App" [
    Container "Menu" [
        Button "btn_new" "New",
        Button "btn_open" "Open",
        Button "btn_close" "Close"],
    Container "Body" [TextBox "textbox_1" "Some text goes here"],
    Container "Footer" []]



countCurrences :: Component -> (Int, Int, Int)
countCurrences (TextBox _ _) = (1, 0, 0)
countCurrences (Button _ _) = (0, 1, 0)
countCurrences (Container _ potomek) =
    let counts = foldr (\p (tb, btn, cont) ->
                        let (tb', btn', cont') = countCurrences p
                        in (tb + tb', btn + btn', cont + cont')
                      ) (0, 0, 1) potomek
    in counts
