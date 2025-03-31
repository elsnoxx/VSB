import csv
import random

def generate_random_data(filename, num_records):
    types = ["Monitor", "PC", "Scanner", "Printer", "Laptop"]
    manufacturers = ["Dell", "Adwantech", "HP", "Lenovo", "Asus"]
    statuses = ["Used", "Damaged", "New", "Refurbished"]

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Záhlaví CSV
        writer.writerow(["ID", "Type", "Manufacturer", "S/N", "Status"])

        for i in range(1, num_records + 1):
            record_type = random.choice(types)
            manufacturer = random.choice(manufacturers)
            serial_number = random.randint(1000000000, 9999999999)  # Náhodné 10místné číslo
            status = random.choice(statuses)

            writer.writerow([i, record_type, manufacturer, serial_number, status])

# Generování 100 náhodných záznamů
generate_random_data("data.csv", 100)