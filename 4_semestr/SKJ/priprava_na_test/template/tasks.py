class TradeException(BaseException):
    pass


class TaxEvaluator:
    """
    Úkol 1

    V roce 2022 probíhal crypto-boom, a kdo si nekoupil NFT nebo alespoň nějakou kryptoměnu, jako
    by nebyl. Někteří investoři nicméně zapomněli, že ze zisků se musí platit daně, a na tyto osoby
    si nyní chce posvítit finanční úřad. Jejich zaměstnanci nicméně nemají dostatek prostředků pro
    analýzu všech nákupů a prodejů kryptoměn. Pomozte jim naimplementováním třídy `TaxEvaluator`,
    která bude počítat výnosy či ztráty obchodů s kryptoměnami, a umožní vyměřit jednotlivým daňovým
    poplatníkům jejich daň.

    Daňová kalkulačka si bude uchovávat informace o tom, kolik mincí jednotlivých kryptoměn nakoupila
    konkrétní osoba, a za jakou cenu je nakoupila. Poté umožní vypočítat celkový zisk pro danou osobu,
    a pokud bude zisk kladný, tak z něj vypočte i finální daň.

    Třída bude poskytovat následující rozhraní:
    ```python
    # Vytvoření daňové kalkulačky. V konstruktoru přijme seznam osob, pro které
    # se budou daně počítat.
    evaluator = TaxEvaluator(["Jan Novák", "Adam Černín", "Markéta Zavadská"])

    # Zaznamená nákup kryptoměny
    # První argument je název osoby
    # Druhý argument je název kryptoměny
    # Třetí argument je cena kryptoměny v době nákupu (celé číslo)
    # Čtvrtý argument je počet nakoupených mincí (celé číslo)
    evaluator.buy_crypto("Jan Novák", "Bitcoin", 890231, 3)

    evaluator.buy_crypto("Adam Černín", "PyCoin", 15, 2)
    evaluator.buy_crypto("Adam Černín", "PyCoin", 25, 3)
    evaluator.buy_crypto("Adam Černín", "PyCoin", 20, 4)

    # Zaznamená prodej kryptoměny
    # První argument je název osoby
    # Druhý argument je název kryptoměny
    # Třetí argument je cena kryptoměny v době prodeje (celé číslo)
    # Čtvrtý argument je počet prodaných mincí (celé číslo)
    #
    # Metoda vrátí zisk, který uživatel získal tímto prodejem (zisk může být i záporný :) ).
    # Více detailů o tom, jak funguje prodej, je poskytnuto níže.
    evaluator.sell_crypto("Adam Černín", "PyCoin", 16, 5) # -10

    # Vrátí současnou celkovou hodnotu všech mincí dané kryptoměny pro konkrétní osobu
    # První argument je název osoby
    # Druhý argument je název kryptoměny
    # Třetí argument je současná cena kryptoměny
    #
    # Adam má celkem 4 mince PyCoinu (9 koupil a 5 prodal), jejich současná hodnota
    # je tedy 4 * 18 = 72.
    evaluator.total_coin_value("Adam Černín", "PyCoin", 18)  # 72

    evaluator.sell_crypto("Adam Černín", "PyCoin", 254, 3)  # 692

    # Spočte daň ze zisků pro danou osobu.
    # Pokud má daná osoba nějaký zisk, tak je daň rovna 15 % celkového získu.
    # Pokud je zisk dané osoby nekladný, tak je daň rovna nule.
    # Počítá se pouze realizovaný zisk, tj. součet všech zisků ze všech uskutečněných prodejů.
    # Daň zaokrouhlete nahoru na celé číslo.
    evaluator.get_tax("Adam Černín")  # 75
    ```

    Jak funguje prodej:
    - Vždy se nejprve prodávají mince nakoupené za nejnižší cenu.
        Příklad:
        Adam nakoupil 2 mince za 15 Kč, 3 mince za 25 Kč a nakonec 4 mince za 20 Kč.
        Nyní prodává 5 mincí za 16 Kč.

        Prodá tedy 2 mince nakoupené za 15 Kč a 3 mince nakoupené za 20 Kč.
        Zbude mu 1 mince nakoupená za 20 Kč a 3 mince nakoupené za 25 Kč.
        Celkový zisk z prodeje bude -10 Kč (nákup stál 2 * 15 + 3 * 20, prodej vynesl 5 * 16).
    - Pokud nedrží osoba v době prodeje dostatek mincí, tak vyvolejte výjimku `TradeException`.

    Pokud některá z výše zmíněných metod obdrží jméno osoby, o které kalkulačka neví,
    vyvolejte výjimku `TradeException`.

    Příklady použití třídy naleznete v testech.
    """
    pass


class TransactionException(BaseException):
    pass


def generate_wallet_info(transaction_path: str):
    """
    Úkol 2

    Kromě samotného zdanění se finanční úřad chce zaměřit také na podezřelé transakce.
    Analytici úřadu chtějí zanalyzovat, kteří daňoví poplatníci si posílali peníze mezi sebou,
    kolik transakcí provedli, a kdo byl nejčastějším příjemcem jejich transakcí. Pomozte jim naimplementováním
    funkce `generate_wallet_info`, která obdrží cestu k souboru se záznamem transakcí kryptoměny
    PyCoin, a vrátí údaje o transakcích jednotlivých krypto peněženek (účtů).

    Soubor obsahuje seznam transakcí, které jsou seřazeny podle toho, v jakém pořadí se provedly.
    Každý řádek souboru obsahuje jednu transakci. Existují dva typy transakcí:
    1) Vytvoření peněz
    Formát: `-> <id> <amount>`
    Tato transakce přidá na účet peněženky `<id>` zadanou částku.
    Tato transakce se počítá jako příchozí transakce pro danou peněženku.

    2) Převod peněz
    Formát: `<src> -> <dst> <amount`
    Tato transakce převede zadanou částku z peněženky `<src>` do peněženky `<dst>`.
    Tato transakce se počítá jako odchozí transakce pro `<src>` a jako příchozí transakce pro `<dst>`.

    Pokud v momentě, kdy je transakce prováděna, nemá peněženka `<src>` dostatečný obnos mincí pro
    provedení transakce, vyvolejte výjimku `TransactionException`.

    Můžete předpokládat, že obsah souboru bude splňovat zadaný formát transakcí.
    Všechny částky budou celá čísla.
    ID peněženek budou obsahovat pouze písmena anglické abecedy a nebudou obsahovat mezery.

    Na ukázky vstupu se můžete podívat v souborech `tx-*.txt`.

    Jakmile zpracujete všechny transakce, tak vraťte pole s údaji pro každou peněženku zmíněnou
    v seznamu transakcí.
    Pole bude seřazeno vzestupně podle ID peněženky.
    Pro každou peněženku vraťte slovník s následujícími klíči:
    ```
    {
        "id": <ID peněženky>,
        "balance": <finální konto peněženky po provedení všech transakcí>,
        "incoming-count": <počet příchozích transakcí peněženky>,
        "outgoing-count": <počet odchozích transakcí peněženky>,
        "most-frequent-target": <ID peněženky, na kterou odešlo nejvíce transakcí z této peněženky>
    }
    ```
    Poznámky k `most-frequent-target`:
    - Pokud by se stalo, že největší počet transakcí odešel na více než jednu peněženku, tak z nich
    vyberte ID peněženky s nejnižším ID (řazeno lexikograficky).
    - Pokud z peněženky neodešla ani jedna odchozí transakce, tak nastavte tento klíč na hodnotu
    `None`.

    Na začátku všechny peněženky začínají s prázdným kontem (tj. mají na kontě 0 mincí).
    """
