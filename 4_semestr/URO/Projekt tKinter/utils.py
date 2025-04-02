import csv


def save_data_to_file(sample_data, filename="data.csv"):
    csvfile = csv.writer(open("data.csv", "w", newline=""))
    try:
        for item in sample_data:
            csvfile.writerow(item)
        print("Soubor 'data.csv' byl úspěšně otevřen a data byla načtena.")
    except FileNotFoundError:
        print("Soubor 'data.csv' nebyl nalezen. Načítání dat selhalo.")
    except KeyError as e:
        print(f"Chyba ve struktuře CSV souboru: {e}")
        
        
def load_data_from_file(filnema="data.csv"):
    sample_data = []
    try:
        with open("data.csv", mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Převod dat na tuple a přidání do sample_data
                sample_data.append((
                    int(row["ID"]),
                    row["Type"],
                    row["Manufacturer"],
                    row["S/N"],
                    row["Status"]
                ))
        return sample_data
    except FileNotFoundError:
        print("Soubor 'data.csv' nebyl nalezen. Načítání dat selhalo.")
    except KeyError as e:
        print(f"Chyba ve struktuře CSV souboru: {e}")