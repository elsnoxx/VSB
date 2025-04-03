import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk, StringVar, messagebox  # Import messagebox
import csv
from tkinter import filedialog

from utils import save_data_to_file, load_data_from_file

Manufacturer = ["", "Dell", "Adwantech", "HP", "Lenovo", "Asus"]
Status = ["", "Used", "Damaged", "New", "Refurbished"]
Type = ["", "Monitor", "PC", "Scanner", "Printer", "Laptop"]
Location = ["", "Brno", "Prague", "Ostrava", "Plzen", "Olomouc"]
basic_fields = ["Type", "Manufacturer", "Price", "Serial Number", "Status"]
additional_fields = ["Location", "Purchase Date", "Size"]

class InventoryApp(ThemedTk):  # Změna z tk.Tk na ThemedTk
    def __init__(self):
        super().__init__()
        self.set_theme("black")
        self.title("Inventory System")
        self.geometry("1150x600")

        # Vytvoření stylů
        style = ttk.Style()
        style.configure("Change.TButton", background="#4CAF50", foreground="white", font=("Arial", 15, "bold"))
        style.configure("Delete.TButton", background="#F44336", foreground="white", font=("Arial", 15, "bold"))
        style.configure("New.TButton", background="#2196F3", foreground="white", font=("Arial", 15, "bold"))
        style.configure("SearchFrame.TFrame", background="#999999", foreground="black")
        style.configure("SearchFrame.TLabel", background="#999999", foreground="black")
        style.configure("Treeview.Heading", background="black", foreground="white", font=("Arial", 10, "bold"))
        style.configure("search_frame.TLabel", background="black", foreground="white", font=("Arial", 10, "bold"))
        style.configure("infoFrame.TLabel", background="black", foreground="white", font=("Arial", 10, "bold"))
        style.configure("Treeview", rowheight=25)
        
        # Nastavení hlavního okna
        self.rowconfigure(0, weight=1)  # Hlavní rámec se roztáhne vertikálně
        self.columnconfigure(0, weight=1)  # Hlavní rámec se roztáhne horizontálně

        # Přidání menu lišty
        self.create_menu()

        # Hlavní rámec
        main_frame = ttk.Frame(self, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        # Nastavení roztažení sloupců a řádků v main_frame
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Tabulka zařízení (levý sloupec)
        tree_frame = ttk.Frame(main_frame)  # Rámeček pro Treeview a Scrollbar
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

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
        side_frame.grid(row=0, column=1, sticky="nsew")

        # Nastavení roztažení side_frame
        side_frame.rowconfigure(1, weight=1)
        side_frame.columnconfigure(0, weight=1)

        # Rámeček pro vyhledávání
        search_frame = ttk.LabelFrame(side_frame, text="Search Options", padding=10, style="search_frame.TLabel")
        search_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        search_frame.configure(style="SearchFrame.TFrame")

        # Filtr podle Serial Number
        ttk.Label(search_frame, text="Serial Number:", style="SearchFrame.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.search_serial = ttk.Entry(search_frame)
        self.search_serial.grid(row=0, column=1, sticky="ew", padx=5, pady=2, columnspan=3)

        # Filtr podle Manufacturer
        ttk.Label(search_frame, text="Manufacturer:", style="SearchFrame.TLabel").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.manufacturer_var = StringVar()
        self.search_manufacturer = ttk.Combobox(search_frame, textvariable=self.manufacturer_var, state="readonly")
        self.search_manufacturer['values'] = Manufacturer
        self.search_manufacturer.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Status
        ttk.Label(search_frame, text="Status:", style="SearchFrame.TLabel").grid(row=2, column=3, sticky="w", padx=5, pady=2)
        self.status_var = StringVar()
        self.search_status = ttk.Combobox(search_frame, textvariable=self.status_var, state="readonly")
        self.search_status['values'] = Status
        self.search_status.grid(row=2, column=4, sticky="ew", padx=5, pady=2)

        # Filtr podle Type
        ttk.Label(search_frame, text="Type:", style="SearchFrame.TLabel").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.type_var = StringVar()
        self.search_type = ttk.Combobox(search_frame, textvariable=self.type_var, state="readonly")
        self.search_type['values'] = Type
        self.search_type.grid(row=3, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Location
        ttk.Label(search_frame, text="Location:", style="SearchFrame.TLabel").grid(row=3, column=3, sticky="w", padx=5, pady=2)
        self.location_var = StringVar()
        self.search_locationa = ttk.Combobox(search_frame, textvariable=self.location_var, state="readonly")
        self.search_locationa['values'] = Location
        self.search_locationa.grid(row=3, column=4, sticky="ew", padx=5, pady=2)

        # Tlačítko pro vyhledávání
        ttk.Button(search_frame, text="Search", style="New.TButton", command=self.search).grid(row=4, column=4, sticky="ew", padx=5, pady=5)

        # Rámeček pro informace o zařízení
        info_frame = ttk.LabelFrame(side_frame, text="Device Information", padding=2, style="infoFrame.TLabel")
        info_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Přidání Notebooku do info_frame
        notebook = ttk.Notebook(info_frame)
        notebook.pack(fill="both", expand=True)

        # Záložka 1: Základní informace
        frame1 = ttk.Frame(notebook, padding=5)
        notebook.add(frame1, text="Basic Info")
        frame1.configure(style="SearchFrame.TFrame")

        # Záložka 2: Další informace
        frame2 = ttk.Frame(notebook, padding=5)
        notebook.add(frame2, text="Additional Info")
        frame2.configure(style="SearchFrame.TFrame")

        # Záložka 3: Obrázky
        frame3 = ttk.Frame(notebook, padding=5)
        notebook.add(frame3, text="Pictures")
        frame3.configure(style="SearchFrame.TFrame")

        # Základní informace (záložka 1)
        self.basic_details = {}
        for i, field in enumerate(basic_fields):
            ttk.Label(frame1, text=field).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.basic_details[field] = ttk.Entry(frame1)
            self.basic_details[field].grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.basic_details[field].configure(state="SearchFrame.TLabel")
            self.basic_details[field].configure(state="readonly")

        # Nastavení roztažení sloupců v gridu
        frame1.columnconfigure(1, weight=1)

        # Další informace (záložka 2)
        self.additional_details = {}
        for i, field in enumerate(additional_fields):
            ttk.Label(frame2, text=field).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.additional_details[field] = ttk.Entry(frame2)
            self.additional_details[field].grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.additional_details[field].configure(state="SearchFrame.TLabel")

        # Nastavení roztažení sloupců v gridu
        frame2.columnconfigure(1, weight=1)

        # Tlačítka vedle sebe
        button_frame = ttk.Frame(side_frame)
        button_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        

        # Tlačítka
        ttk.Button(button_frame, text="Change status", style="Change.TButton", command=self.changeStatus).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Delete Device", style="Delete.TButton", command=self.deleteDevice).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(button_frame, text="New Device", style="New.TButton", command=self.addNew).grid(row=0, column=2, padx=5, pady=5, sticky="ew")

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
                    for field in basic_fields:
                        if field in self.basic_details:
                            self.basic_details[field].configure(state="normal")
                            self.basic_details[field].delete(0, tk.END)
                            self.basic_details[field].insert(0, item.get(field, ""))
                            self.basic_details[field].configure(state="readonly")

                    # Naplnění dalších informací
                    for field in additional_fields:
                        if field in self.additional_details:
                            self.additional_details[field].configure(state="normal")
                            self.additional_details[field].delete(0, tk.END)
                            self.additional_details[field].insert(0, item.get(field, ""))
                            self.additional_details[field].configure(state="readonly")
                    break

    def show_about(self):
        messagebox.showinfo("About", "Inventory System v1.0\nCreated by FIC0024")

    def changeStatus(self):
        

        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No device selected!")
            return

        # Získání dat vybraného zařízení
        values = self.tree.item(selected_item, "values")

        # Vytvoření popup okna
        popup = tk.Toplevel(self)
        self.set_theme("black")
        popup.title("Edit Device")
        popup.geometry("400x300")

        style = ttk.Style()
        style.configure("SearchFrame.TFrame", background="#999999")
        popup.configure(style="SearchFrame.TFrame")

        # Pole pro zobrazení a úpravu dat
        fields = ["Type", "Manufacturer", "Serial Number", "Status", "Location"]
        combobox_data = {
            "Type": Type,
            "Manufacturer": Manufacturer,
            "Status": Status,
            "Location": Location
        }
        entries = {}

        for i, field in enumerate(fields):
            ttk.Label(popup, text=field).grid(row=i, column=0, padx=10, pady=5, sticky="w")
            if field in combobox_data:
                # Použití Comboboxu pro pole s předdefinovanými hodnotami
                combobox = ttk.Combobox(popup, values=combobox_data[field], state="readonly")
                combobox.set(values[i + 1])  # Nastavení aktuální hodnoty
                combobox.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = combobox
            else:
                # Použití Entry pro ostatní pole
                entry = ttk.Entry(popup)
                entry.insert(0, values[i + 1])  # Přeskočíme ID
                entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = entry

        # Funkce pro uložení změn
        def save_changes():
            new_values = [values[0]] + [entries[field].get() for field in fields]
            self.tree.item(selected_item, values=new_values)
            print("Device has been updated:", new_values)
            popup.destroy()

        # Tlačítka pro uložení nebo zrušení
        ttk.Button(popup, text="Save", command=save_changes).grid(row=len(fields), column=0, padx=10, pady=10, sticky="ew")
        ttk.Button(popup, text="Cancel", command=popup.destroy).grid(row=len(fields), column=1, padx=10, pady=10, sticky="ew")

        popup.columnconfigure(1, weight=1)

    def addNew(self):
        # Vytvoření popup okna
        popup = tk.Toplevel(self)
        popup.title("Add New Device")
        popup.geometry("400x300")

        # Pole pro zadání dat
        fields = ["Type", "Manufacturer", "Serial Number", "Status", "Location"]
        entries = {}

        for i, field in enumerate(fields):
            ttk.Label(popup, text=field).grid(row=i, column=0, padx=10, pady=5, sticky="w")
            entry = ttk.Entry(popup)
            entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
            entries[field] = entry

        # Funkce pro uložení dat
        def save_data():
            new_id = 1
            for item in self.tree.get_children():
                existing_id = int(self.tree.item(item, "values")[0])
                if existing_id >= new_id:
                    new_id = existing_id + 1
            print("New ID:", new_id)
            values = [new_id] + [entries[field].get() for field in fields]
            self.tree.insert("", "end", values=values)
            print("New device has been added:", values)
            popup.destroy()

        # Tlačítka pro uložení nebo zrušení
        ttk.Button(popup, text="Save", command=save_data).grid(row=len(fields), column=0, padx=10, pady=10, sticky="ew")
        ttk.Button(popup, text="Cancel", command=popup.destroy).grid(row=len(fields), column=1, padx=10, pady=10, sticky="ew")

        popup.columnconfigure(1, weight=1)

    def deleteDevice(self):
        selected_items = self.tree.selection()
        for selected_item in selected_items:
            values = self.tree.item(selected_item, "values")
            self.tree.delete(selected_item)
            print("Deleting device:", values)
        
if __name__ == "__main__":
    app = InventoryApp()
    app.mainloop()
