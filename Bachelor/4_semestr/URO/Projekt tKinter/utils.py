import csv
import json


def save_data_to_file(sample_data, filename="data.json"):
    """
    Uloží data do JSON souboru.
    """
    try:
        with open(filename, mode="w", encoding="utf-8") as file:
            json.dump(sample_data, file, indent=4)
        print(f"Soubor '{filename}' byl úspěšně uložen.")
    except Exception as e:
        print(f"Chyba při ukládání dat do souboru '{filename}': {e}")


def load_data_from_file(filename="data.json"):
    """
    Načte data z JSON souboru a vrátí je jako seznam slovníků.
    """
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            sample_data = json.load(file)
        print(f"Soubor '{filename}' byl úspěšně načten.")
        return sample_data
    except FileNotFoundError:
        print(f"Soubor '{filename}' nebyl nalezen. Načítání dat selhalo.")
        return []
    except json.JSONDecodeError as e:
        print(f"Chyba při dekódování JSON souboru '{filename}': {e}")
        return []