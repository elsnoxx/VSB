import tkinter as tk
from tkinter import ttk, StringVar
import csv
from tkinter import filedialog

from utils import save_data_to_file, load_data_from_file

class InventoryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Inventory System")
        self.geometry("800x600")
        
        # Přidání menu lišty
        self.create_menu()

        # Hlavní rámec
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill="both", expand=True)

        # Tabulka zařízení
        self.tree = ttk.Treeview(main_frame, columns=("ID", "Type", "Manufacturer", "S/N", "Status"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)
        self.tree.pack(side="left", fill="both", expand=True)

        # Přidání události pro výběr položky
        self.tree.bind("<<TreeviewSelect>>", self.on_item_selected)

        # Panel s detaily a vyhledáváním
        side_frame = ttk.Frame(main_frame, padding=5)
        side_frame.pack(side="right", fill="y")

        ttk.Label(side_frame, text="Search Settings:").pack(anchor="w")

        # Filtr podle Serial Number
        ttk.Label(side_frame, text="Serial Number:").pack(anchor="w")
        self.search_serial = ttk.Entry(side_frame)
        self.search_serial.pack(fill="x")

        # Filtr podle Manufacturer
        ttk.Label(side_frame, text="Manufacturer:").pack(anchor="w")
        self.manufacturer_var = StringVar()
        self.search_manufacturer = ttk.Combobox(side_frame, textvariable=self.manufacturer_var, state="readonly")
        self.search_manufacturer['values'] = ["", "Dell", "Adwantech"]  # Přidat další hodnoty podle potřeby
        self.search_manufacturer.pack(fill="x")

        # Filtr podle Status
        ttk.Label(side_frame, text="Status:").pack(anchor="w")
        self.status_var = StringVar()
        self.search_status = ttk.Combobox(side_frame, textvariable=self.status_var, state="readonly")
        self.search_status['values'] = ["", "Used", "Damaged"]  # Přidat další hodnoty podle potřeby
        self.search_status.pack(fill="x")

        # Filtr podle Type
        ttk.Label(side_frame, text="Type:").pack(anchor="w")
        self.type_var = StringVar()
        self.search_type = ttk.Combobox(side_frame, textvariable=self.type_var, state="readonly")
        self.search_type['values'] = ["", "Monitor", "PC", "Scanner"]  # Přidat další hodnoty podle potřeby
        self.search_type.pack(fill="x")

        # Tlačítko pro vyhledávání
        ttk.Button(side_frame, text="Search", command=self.search).pack(fill="x")

        # Detailní informace
        ttk.Label(side_frame, text="More Information:").pack(anchor="w")
        self.info_frame = ttk.LabelFrame(side_frame, text="Device Details", padding=5)
        self.info_frame.pack(fill="both", expand=True)

        self.details = {}
        for field in ["Type", "User", "Size", "Price", "S/N", "Location", "Purchase date", "Manufacturer", "Note"]:
            ttk.Label(self.info_frame, text=field).pack(anchor="w")
            self.details[field] = ttk.Entry(self.info_frame)
            self.details[field].pack(fill="x")

        # Ovládací tlačítka
        ttk.Button(side_frame, text="Change status").pack(fill="x")
        ttk.Button(side_frame, text="Delete Device").pack(fill="x")
        ttk.Button(side_frame, text="New Device").pack(fill="x")

        # Ukázková data
        self.load_data()

    def load_data(self):
        self.sample_data = load_data_from_file()
        print(self.sample_data)
        for item in self.sample_data:
            self.tree.insert("", "end", values=item)

    def create_menu(self):
        # Vytvoření menu lišty
        menu_bar = tk.Menu(self)

        # Menu "File"
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save As", command=self.save_as_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Menu "Help"
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

        # Nastavení menu lišty
        self.config(menu=menu_bar)
    
    def open_file(self):
        filpath = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Open file"
        )
        self.sample_data = load_data_from_file(filpath)
        
        for item in self.sample_data:
            self.tree.insert("", "end", values=item)

    def save_file(self):
        save_data_to_file(self.sample_data)

    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save as"
        )
        save_data_to_file(file_path, self.sample_data)
        
    def show_about(self):
        tk.Message("About", "Inventory System v1.0\nCreated by Your Name")
        
    def search(self):
        # Získání hodnot z filtrů
        serial = self.search_serial.get().strip()
        manufacturer = self.manufacturer_var.get().strip()
        status = self.status_var.get().strip()
        device_type = self.type_var.get().strip()

        # Vymazání aktuálního obsahu tabulky
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Filtrování dat
        for item in self.sample_data:
            if serial and serial not in item[3]:
                continue
            if manufacturer and manufacturer != item[2]:
                continue
            if status and status != item[4]:
                continue
            if device_type and device_type != item[1]:
                continue
            self.tree.insert("", "end", values=item)

    def on_item_selected(self, event):
        # Získání vybrané položky
        selected_item = self.tree.selection()
        if selected_item:
            item_values = self.tree.item(selected_item[0], "values")
            # Naplnění detailních informací
            fields = ["Type", "User", "Size", "Price", "S/N", "Location", "Purchase date", "Manufacturer", "Note"]
            for i, field in enumerate(fields):
                if i < len(item_values):
                    self.details[field].delete(0, tk.END)
                    self.details[field].insert(0, item_values[i])


if __name__ == "__main__":
    app = InventoryApp()
    app.mainloop()
