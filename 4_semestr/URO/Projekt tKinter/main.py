import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk, StringVar, messagebox  # Import messagebox
import csv
from tkinter import filedialog

from utils import save_data_to_file, load_data_from_file

class InventoryApp(ThemedTk):  # Změna z tk.Tk na ThemedTk
    def __init__(self):
        super().__init__()
        self.set_theme("black")  # Nastavení tématu "black"
        self.title("Inventory System")
        self.geometry("1150x600")

        # Vytvoření stylů pro tlačítka
        style = ttk.Style()
        style.configure("Change.TButton", background="#4CAF50", foreground="white")  # Zelené tlačítko
        style.configure("Delete.TButton", background="#F44336", foreground="white")  # Červené tlačítko
        style.configure("New.TButton", background="#2196F3", foreground="white")     # Modré tlačítko
        style.configure("SearchFrame.TFrame", background="green")  # Styl pro SearchFrame
        
        # Nastavení hlavního okna
        self.rowconfigure(0, weight=1)  # Hlavní rámec se roztáhne vertikálně
        self.columnconfigure(0, weight=1)  # Hlavní rámec se roztáhne horizontálně

        # Přidání menu lišty
        self.create_menu()

        # Hlavní rámec
        main_frame = ttk.Frame(self, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")  # Použití grid místo pack

        # Nastavení roztažení sloupců a řádků v main_frame
        main_frame.columnconfigure(0, weight=3)  # Levý sloupec (Treeview) zabírá více místa
        main_frame.columnconfigure(1, weight=1)  # Pravý sloupec (side_frame)
        main_frame.rowconfigure(0, weight=1)  # Oba sloupce se roztáhnou vertikálně

        # Tabulka zařízení (levý sloupec)
        tree_frame = ttk.Frame(main_frame)  # Rámeček pro Treeview a Scrollbar
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))  # Použití grid místo pack

        # Nastavení roztažení Treeview
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        self.tree = ttk.Treeview(tree_frame, columns=("ID", "Type", "Manufacturer", "Serial Number", "Status", "Location"), show="headings")
        for col in self.tree["columns"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")

        # Přidání Scrollbaru
        tree_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scrollbar.set)

        # Umístění Treeview a Scrollbaru
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scrollbar.grid(row=0, column=1, sticky="ns")

        # Přidání události pro výběr položky
        self.tree.bind("<<TreeviewSelect>>", self.on_item_selected)

        # Panel s detaily a vyhledáváním (pravý sloupec)
        side_frame = ttk.Frame(main_frame, padding=5)
        side_frame.grid(row=0, column=1, sticky="nsew")  # Použití grid místo pack

        # Nastavení roztažení side_frame
        side_frame.rowconfigure(1, weight=1)  # Info frame se roztáhne
        side_frame.columnconfigure(0, weight=1)

        # Rámeček pro vyhledávání
        search_frame = ttk.LabelFrame(side_frame, text="Search Options", padding=10)
        search_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        # search_frame.configure(style="SearchFrame.TFrame")

        # Filtr podle Serial Number
        ttk.Label(search_frame, text="Serial Number:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.search_serial = ttk.Entry(search_frame)
        self.search_serial.grid(row=0, column=1, sticky="ew", padx=5, pady=2, columnspan=3)

        # Filtr podle Manufacturer
        ttk.Label(search_frame, text="Manufacturer:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.manufacturer_var = StringVar()
        self.search_manufacturer = ttk.Combobox(search_frame, textvariable=self.manufacturer_var, state="readonly")
        self.search_manufacturer['values'] = ["", "Dell", "Adwantech", "HP", "Lenovo", "Asus"]
        self.search_manufacturer.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Status
        ttk.Label(search_frame, text="Status:").grid(row=2, column=3, sticky="w", padx=5, pady=2)
        self.status_var = StringVar()
        self.search_status = ttk.Combobox(search_frame, textvariable=self.status_var, state="readonly")
        self.search_status['values'] = ["", "Used", "Damaged", "New", "Refurbished"]
        self.search_status.grid(row=2, column=4, sticky="ew", padx=5, pady=2)

        # Filtr podle Type
        ttk.Label(search_frame, text="Type:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.type_var = StringVar()
        self.search_type = ttk.Combobox(search_frame, textvariable=self.type_var, state="readonly")
        self.search_type['values'] = ["", "Monitor", "PC", "Scanner", "Printer", "Laptop"]
        self.search_type.grid(row=3, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Location
        ttk.Label(search_frame, text="Location:").grid(row=3, column=3, sticky="w", padx=5, pady=2)
        self.location_var = StringVar()
        self.search_locationa = ttk.Combobox(search_frame, textvariable=self.location_var, state="readonly")
        self.search_locationa['values'] = ["", "Brno", "Prague", "Ostrava", "Plzen", "Olomouc"]
        self.search_locationa.grid(row=3, column=4, sticky="ew", padx=5, pady=2)

        # Tlačítko pro vyhledávání
        ttk.Button(search_frame, text="Search", style="New.TButton", command=self.search).grid(row=4, column=4, sticky="ew", padx=5, pady=5)


        # Rámeček pro informace o zařízení
        info_frame = ttk.LabelFrame(side_frame, text="Device Information", padding=2)
        info_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Přidání Notebooku do info_frame
        notebook = ttk.Notebook(info_frame)
        notebook.pack(fill="both", expand=True)

        # Záložka 1: Základní informace
        frame1 = ttk.Frame(notebook, padding=5)
        notebook.add(frame1, text="Basic Info")

        # Záložka 2: Další informace
        frame2 = ttk.Frame(notebook, padding=5)
        notebook.add(frame2, text="Additional Info")

        # Záložka 3: Obrázky
        frame3 = ttk.Frame(notebook, padding=5)
        notebook.add(frame3, text="Pictures")

        # Základní informace (záložka 1)
        basic_fields = ["Type", "Manufacturer", "Price", "Serial Number"]
        self.basic_details = {}
        for field in basic_fields:
            ttk.Label(frame1, text=field).pack(anchor="w")
            self.basic_details[field] = ttk.Entry(frame1)
            self.basic_details[field].pack(fill="x")

        # Další informace (záložka 2)
        additional_fields = ["Location", "Purchase Date", "Size"]
        self.additional_details = {}
        for field in additional_fields:
            ttk.Label(frame2, text=field).pack(anchor="w")
            self.additional_details[field] = ttk.Entry(frame2)
            self.additional_details[field].pack(fill="x")

        # Tlačítka vedle sebe
        button_frame = ttk.Frame(side_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        

        # Tlačítka
        ttk.Button(button_frame, text="Change status", style="Change.TButton").grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Delete Device", style="Delete.TButton").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="New Device", style="New.TButton").grid(row=0, column=2, padx=5, pady=5, sticky="ew")

        # Nastavení roztažení sloupců a řádků
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        # Ukázková data
        self.load_data()

    def load_data(self):
        self.sample_data = load_data_from_file("data.json")
        # Vymazání aktuálního obsahu Treeview
        for item in self.tree.get_children():
            self.tree.delete(item)
        # Zobrazení dat v Treeview
        for item in self.sample_data:
            self.tree.insert("", "end", values=(
                item.get("ID", ""),
                item.get("Type", ""),
                item.get("Manufacturer", ""),
                item.get("Serial Number", ""),
                item.get("Status", ""),
                item.get("Location", "")
            ))

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
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Open file"
        )
        if filepath:
            self.sample_data = load_data_from_file(filepath)
            # Vymazání aktuálního obsahu Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            # Zobrazení dat v Treeview
            for item in self.sample_data:
                self.tree.insert("", "end", values=(
                    item.get("ID", ""),
                    item.get("Type", ""),
                    item.get("Manufacturer", ""),
                    item.get("Serial Number", ""),
                    item.get("Status", ""),
                    item.get("Location", "")
                ))

    def save_as_file(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save as"
        )
        if file_path:
            # Převod dat z Treeview zpět do seznamu slovníků
            self.sample_data = []
            for item in self.tree.get_children():
                values = self.tree.item(item, "values")
                self.sample_data.append({
                    "ID": values[0],
                    "Type": values[1],
                    "Manufacturer": values[2],
                    "Serial Number": values[3],
                    "Status": values[4],
                    "Location": values[5]
                })
            # Uložení dat do zvoleného souboru
            save_data_to_file(self.sample_data, file_path)

    def save_file(self):
        # Převod dat z Treeview zpět do seznamu slovníků
        self.sample_data = []
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            self.sample_data.append({
                "ID": values[0],
                "Type": values[1],
                "Manufacturer": values[2],
                "Serial Number": values[3],
                "Status": values[4],
                "Location": values[5]
            })
        # Uložení dat do JSON souboru
        save_data_to_file(self.sample_data)

    def search(self, return_results=False):
        """
        Vyhledávání dat podle zadaných filtrů.
        Pokud `return_results` je True, vrátí výsledky jako seznam místo zobrazení v Treeview.
        """
        # Získání hodnot z filtrů
        serial = self.search_serial.get().strip()
        manufacturer = self.manufacturer_var.get().strip()
        status = self.status_var.get().strip()
        device_type = self.type_var.get().strip()
        location = self.location_var.get().strip()

        # Filtrování dat
        filtered_data = []
        for item in self.sample_data:
            if serial and serial not in item.get("Serial Number", ""):
                continue
            if manufacturer and manufacturer != item.get("Manufacturer", ""):
                continue
            if status and status != item.get("Status", ""):
                continue
            if device_type and device_type != item.get("Type", ""):
                continue
            if location and location != item.get("Location", ""):
                continue
            filtered_data.append(item)

        # Pokud je `return_results` True, vrátí výsledky jako seznam
        if return_results:
            return filtered_data

        # Jinak zobrazí výsledky v Treeview
        # Vymazání aktuálního obsahu tabulky
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Zobrazení filtrovaných dat (pouze vybrané sloupce)
        for item in filtered_data:
            self.tree.insert("", "end", values=(
                item.get("ID", ""),
                item.get("Type", ""),
                item.get("Manufacturer", ""),
                item.get("Serial Number", ""),
                item.get("Status", ""),
                item.get("Location", "")
            ))

    def on_item_selected(self, event):
        # Získání vybrané položky
        selected_item = self.tree.selection()
        if selected_item:
            # Získání hodnot z Treeview
            tree_values = self.tree.item(selected_item[0], "values")
            selected_id = tree_values[0]  # Předpokládáme, že "ID" je první sloupec

            # Vyhledání odpovídajícího záznamu v sample_data
            for item in self.sample_data:
                if str(item.get("ID", "")) == str(selected_id):
                    # Naplnění základních informací
                    basic_fields = ["Type", "Manufacturer", "Price", "Serial Number"]
                    for field in basic_fields:
                        if field in self.basic_details:
                            self.basic_details[field].delete(0, tk.END)
                            self.basic_details[field].insert(0, item.get(field, ""))

                    # Naplnění dalších informací
                    additional_fields = ["Location", "Purchase Date", "Size"]
                    for field in additional_fields:
                        if field in self.additional_details:
                            self.additional_details[field].delete(0, tk.END)
                            self.additional_details[field].insert(0, item.get(field, ""))
                    break

    def show_about(self):
        messagebox.showinfo("About", "Inventory System v1.0\nCreated by FIC0024")


if __name__ == "__main__":
    app = InventoryApp()
    app.mainloop()
