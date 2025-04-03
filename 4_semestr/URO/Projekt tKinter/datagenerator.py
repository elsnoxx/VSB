import json
from datetime import datetime, timedelta
import random
import os

def generate_random_data(filename, num_records):
    types = ["Monitor", "PC", "Scanner", "Printer", "Laptop"]
    manufacturers = ["Dell", "Adwantech", "HP", "Lenovo", "Asus"]
    statuses = ["Used", "Damaged", "New", "Refurbished"]
    locations = ["Brno", "Prague", "Ostrava", "Plzen", "Olomouc"]
    sizes = ["Small", "Medium", "Large", "Extra Large"]  # Možné velikosti produktů

    # Rozsah pro generování náhodného data (např. posledních 5 let)
    start_date = datetime.now() - timedelta(days=5 * 365)
    end_date = datetime.now()

    # Smazání původního souboru, pokud existuje
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Soubor {filename} byl smazán.")

    print(f"Generating {num_records} random records", end="")

    # Generování dat
    data = []
    for i in range(1, num_records + 1):
        print(".", end="")
        record_type = random.choice(types)
        manufacturer = random.choice(manufacturers)
        serial_number = random.randint(1000000000, 9999999999)  # Náhodné 10místné číslo
        status = random.choice(statuses)
        location = random.choice(locations)
        size = random.choice(sizes)
        price = random.randint(1000, 50000)

        # Generování náhodného data nákupu
        random_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        purchase_date = random_date.strftime("%Y-%m-%d")  # Formátování data jako "YYYY-MM-DD"

        # Přidání záznamu do seznamu
        data.append({
            "ID": i,
            "Type": record_type,
            "Manufacturer": manufacturer,
            "Price": price,
            "Serial Number": serial_number,
            "Status": status,
            "Location": location,
            "Size": size,
            "Purchase Date": purchase_date
        })

    # Uložení dat do JSON souboru
    with open(filename, mode="w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    print("\nData generation complete.")

# Generování 100 náhodných záznamů
generate_random_data("data.json", 100)