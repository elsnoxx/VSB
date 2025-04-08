import tkinter as tk
from ttkthemes import ThemedTk
from tkinter import ttk, StringVar, messagebox  # Import messagebox
from tkinter import filedialog
from PIL import Image, ImageTk
import os

from utils import save_data_to_file, load_data_from_file

Manufacturer = ["", "Dell", "Adwantech", "HP", "Lenovo", "Asus"]
Status = ["", "Used", "Damaged", "New", "Refurbished"]
Type = ["", "Monitor", "PC", "Scanner", "Printer", "Laptop"]
Location = ["", "Brno", "Prague", "Ostrava", "Plzen", "Olomouc"]
basic_fields = ["Type", "Manufacturer", "Price", "Serial Number", "Status"]
additional_fields = ["Location", "Purchase Date", "Size"]

fields_frame1 = ["Type", "Manufacturer", "Price", "Serial Number", "Status", "Location", "Purchase Date", "Size"]

class InventoryApp(ThemedTk):  # Změna z tk.Tk na ThemedTk
    def __init__(self):
        super().__init__()
        self.set_theme("black")
        self.title("Inventory System")
        self.geometry("1150x600")

        # Vytvoření stylů
        style = ttk.Style()
        style.configure("Change.TButton", background="#333333", foreground="white", font=("Arial", 10, "bold"), anchor="center")
        style.configure("Search.TButton", background="#333333", foreground="white", font=("Arial", 10, "bold"), anchor="center")
        style.configure("Delete.TButton", background="#666666", foreground="white", font=("Arial", 15, "bold"))
        style.configure("New.TButton", background="#666666", foreground="white", font=("Arial", 15, "bold"))
        style.configure("SearchFrame.TFrame", background="#999999", foreground="black")
        style.configure("SearchFrame.TLabel", background="#999999", foreground="black", anchor="center", font=("Arial", 9, "bold"))
        style.configure("Treeview.Heading", background="black", foreground="white", font=("Arial", 10, "bold"))
        style.configure("search_frame.TLabel", background="black", foreground="white", font=("Arial", 10, "bold"))
        style.configure("infoFrame.TLabel", background="black", foreground="white", font=("Arial", 15, "bold"))
        style.configure("infoFrame.TLabelFrame", background="#333333", foreground="white", font=("Arial", 15, "bold"))
        style.layout("infoFrame.TLabelFrame", [
            ("LabelFrame.border", {"sticky": "nswe", "children": [
                ("LabelFrame.padding", {"sticky": "nswe", "children": [
                    ("LabelFrame.label", {"sticky": "w"})
                ]})
            ]})
        ])
        style.configure("Treeview", rowheight=25)
        style.configure("SearchFrame.Title.TLabel", background="#333333", foreground="white", font=("Arial", 10, "bold"))
        style.configure("SearchFrame.TLabelFrame", background="#333333", foreground="white", font=("Arial", 10, "bold"))
        
        # Vytvoření stylu pro search_frame.TLabelFrame
        style.configure("SearchFrame.TLabelFrame", background="#333333", foreground="white", font=("Arial", 10, "bold"))
        style.layout("SearchFrame.TLabelFrame", [
            ("LabelFrame.border", {"sticky": "nswe", "children": [
                ("LabelFrame.padding", {"sticky": "nswe", "children": [
                    ("LabelFrame.label", {"sticky": "w"})
                ]})
            ]})
        ])

        # Nastavení hlavního okna
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

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

        # Rámeček pro vyhledávání (hlavní rámec s nadpisem)
        search_frame_main = ttk.Frame(side_frame, padding=(10, 10), style="SearchFrame.TLabelFrame")
        search_frame_main.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # Nastavení roztažení search_frame v rámci search_frame_main
        search_frame_main.rowconfigure(1, weight=1)
        search_frame_main.columnconfigure(0, weight=1)

        # Nadpis ve search_frame_main
        title_label = ttk.Label(search_frame_main, text="Search Options:", style="SearchFrame.Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 5))

        # Vnitřní rámec pro obsah vyhledávání
        search_frame = ttk.Frame(search_frame_main, padding=(10, 10), style="SearchFrame.TFrame")
        search_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        

        # Filtr podle Serial Number
        ttk.Label(search_frame, text="Serial Number:", style="SearchFrame.TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.search_serial = ttk.Entry(search_frame)
        self.search_serial.grid(row=0, column=1, sticky="ew", padx=5, pady=2, columnspan=3)

        # Filtr podle Manufacturer
        ttk.Label(search_frame, text="Manufacturer:", style="SearchFrame.TLabel").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.manufacturer_var = StringVar()
        self.search_manufacturer = ttk.Combobox(search_frame, textvariable=self.manufacturer_var, state="readonly")
        self.search_manufacturer['values'] = Manufacturer
        self.search_manufacturer.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Status
        ttk.Label(search_frame, text="Status:", style="SearchFrame.TLabel").grid(row=1, column=3, sticky="w", padx=5, pady=2)
        self.status_var = StringVar()
        self.search_status = ttk.Combobox(search_frame, textvariable=self.status_var, state="readonly")
        self.search_status['values'] = Status
        self.search_status.grid(row=1, column=4, sticky="ew", padx=5, pady=2)

        # Filtr podle Type
        ttk.Label(search_frame, text="Type:", style="SearchFrame.TLabel").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.type_var = StringVar()
        self.search_type = ttk.Combobox(search_frame, textvariable=self.type_var, state="readonly")
        self.search_type['values'] = Type
        self.search_type.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # Filtr podle Location
        ttk.Label(search_frame, text="Location:", style="SearchFrame.TLabel").grid(row=2, column=3, sticky="w", padx=5, pady=2)
        self.location_var = StringVar()
        self.search_locationa = ttk.Combobox(search_frame, textvariable=self.location_var, state="readonly")
        self.search_locationa['values'] = Location
        self.search_locationa.grid(row=2, column=4, sticky="ew", padx=5, pady=2)

        # Nastavení všech řádků a sloupců v gridu
        for i in range(search_frame.grid_size()[1]):  # Počet řádků
            search_frame.rowconfigure(i, weight=1)

        for j in range(search_frame.grid_size()[0]):  # Počet sloupců
            search_frame.columnconfigure(j, weight=2)

        # Tlačítko pro vyhledávání
        ttk.Button(search_frame, text="Search", style="Search.TButton", command=self.search).grid(row=3, column=4, sticky="ew", padx=5, pady=5)

        # Rámeček pro informace o zařízení
        info_frame_main = ttk.Frame(side_frame, padding=(5, 10, 5, 5), style="infoFrame.TLabelFrame")
        info_frame_main.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Nadpis ve info_frame_main
        title_label = ttk.Label(info_frame_main, text="Device Information:", style="SearchFrame.Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 5))

        # Vnitřní rámec pro obsah informací
        info_frame = ttk.Frame(info_frame_main, style="infoFrame.TFrame")
        info_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Nastavení roztažení info_frame v rámci info_frame_main
        info_frame_main.rowconfigure(1, weight=1)
        info_frame_main.columnconfigure(0, weight=1)

        # Přidání Notebooku do info_frame
        notebook = ttk.Notebook(info_frame)
        notebook.grid(row=0, column=0, sticky="nsew", columnspan=2)

        # Nastavení roztažení obsahu uvnitř info_frame
        info_frame.rowconfigure(0, weight=1)
        info_frame.columnconfigure(0, weight=1)

        # Záložka 1: Základní informace
        frame1 = ttk.Frame(notebook, padding=5)
        notebook.add(frame1, text="Basic Info")
        frame1.configure(style="SearchFrame.TFrame")

        # Záložka 2: Další informace
        frame2 = ttk.Frame(notebook, padding=5)
        notebook.add(frame2, text="Pictures")
        frame2.configure(style="SearchFrame.TFrame")



        # Základní informace (záložka 1)
        self.basic_details = {}
        for i, field in enumerate(fields_frame1):
            ttk.Label(frame1, text=field, style="SearchFrame.TLabel").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            self.basic_details[field] = ttk.Entry(frame1)
            self.basic_details[field].grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.basic_details[field].configure(state="readonly")

        # Nastavení roztažení sloupců v gridu
        frame1.columnconfigure(1, weight=1)

        ttk.Button(frame1, text="Edit", style="Change.TButton", command=self.changeStatus).grid(row=10, column=2, padx=5, pady=5)

        # Pictures (záložka 3)
        self.image_label = tk.Label(frame2, background="#999999")
        self.image_label.pack(padx=10, pady=10)
        # Tlačítko pro změnu obrázku
        self.change_image_button = ttk.Button(frame2, text="Change Image", style="Change.TButton", command=self.change_image)
        self.change_image_button.pack(padx=10, pady=10, side="right")

        # Nastavení roztažení sloupců v gridu
        frame2.columnconfigure(1, weight=1)

        # Tlačítka vedle sebe
        ttk.Button(info_frame, text="New Device", style="New.TButton", command=self.addNew).grid(row=1, column=0, padx=5, pady=5, sticky="e")
        ttk.Button(info_frame, text="Delete Device", style="Delete.TButton", command=self.deleteDevice).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

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

        # Nastavení barev pro menu
        self.option_add("*Menu.background", "#333333")
        self.option_add("*Menu.foreground", "white")
        self.option_add("*Menu.activeBackground", "#4CAF50")
        self.option_add("*Menu.activeForeground", "white")
        self.option_add("*Menu.font", ("Arial", 10))

        # Menu "File"
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Open", command=self.open_file)
        file_menu.add_command(label="Save As", command=self.save_as_file)
        file_menu.add_command(label="Save", command=self.save_file)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        # Menu "Edit"
        edit_menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Change Status", command=self.changeStatus)
        edit_menu.add_command(label="Add New Device", command=self.addNew)
        edit_menu.add_command(label="Delete Device", command=self.deleteDevicePopup)
        edit_menu.add_command(label="Change Image", command=self.change_image)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

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
                    for field in fields_frame1:
                        if field in self.basic_details:
                            self.basic_details[field].configure(state="normal")
                            self.basic_details[field].delete(0, tk.END)
                            self.basic_details[field].insert(0, item.get(field, ""))
                            self.basic_details[field].configure(state="readonly")

                    # Načtení a zobrazení obrázku
                    image_path = item.get("Image Path", "")
                    if os.path.exists(image_path):
                        image = Image.open(image_path)
                        image = image.resize((300, 200))  # Změna velikosti obrázku
                        photo = ImageTk.PhotoImage(image)
                        self.image_label.configure(image=photo)
                        self.image_label.image = photo  # Uložení reference na obrázek
                    else:
                        self.image_label.configure(image="", text="No Image Available")
                        self.image_label.image = None

                    break

    def show_about(self):
        messagebox.showinfo("About", "Inventory System v1.0\nCreated by FIC0024")

    def changeStatus(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Warning", "No device selected!")
            return

        # Získání hodnot z Treeview
        tree_values = self.tree.item(selected_item[0], "values")
        selected_id = tree_values[0]  # Předpokládáme, že "ID" je první sloupec

        # Vyhledání odpovídajícího záznamu v sample_data
        selected_data = None
        for item in self.sample_data:
            if str(item.get("ID", "")) == str(selected_id):
                selected_data = item
                break

        if not selected_data:
            messagebox.showerror("Error", "No data found for the selected device!")
            return

        # Vytvoření popup okna
        popup = tk.Toplevel(self)
        popup.title("Edit Device")
        popup.geometry("400x400")
        popup.configure(bg="#333333")  # Tmavé pozadí popup okna
        popup.resizable(False, False)

        # Nastavení gridu popup okna
        popup.rowconfigure(0, weight=1)  # Nadpis
        popup.rowconfigure(1, weight=1)  # Hlavní rámec
        popup.rowconfigure(2, weight=1)  # Tlačítka
        popup.columnconfigure(0, weight=1)

        # Definice stylu pro hlavní rámec
        style = ttk.Style()
        style.configure("Light.TFrame", background="#999999")  # Světlejší pozadí
        style.configure("Dark.TFrame", background="#333333")  # Tmavší pozadí
        style.configure("Popup.TLabel", background="#333333", foreground="white", font=("Arial", 14, "bold"))
        style.configure("main_frame.TLabel", background="#999999", foreground="black", font=("Arial", 10))

        # Nadpis
        title_label = ttk.Label(popup, text="Edit Device", style="Popup.TLabel", anchor="center")
        title_label.grid(row=0, column=0, pady=(10, 10), padx=10, sticky="nsew")

        # Hlavní rámec
        main_frame = ttk.Frame(popup, padding=(10, 10), style="Light.TFrame")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

        # Pole pro úpravu dat
        combobox_data = {
            "Type": Type,
            "Manufacturer": Manufacturer,
            "Status": Status,
            "Location": Location
        }
        entries = {}

        for i, field in enumerate(fields_frame1):
            ttk.Label(main_frame, text=field, style="main_frame.TLabel").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            if field in combobox_data:
                # Použití Comboboxu pro pole s předdefinovanými hodnotami
                combobox = ttk.Combobox(main_frame, values=combobox_data[field], state="readonly")
                combobox.set(selected_data.get(field, ""))  # Nastavení aktuální hodnoty
                combobox.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = combobox
            else:
                # Použití Entry pro ostatní pole
                entry = ttk.Entry(main_frame)
                entry.insert(0, selected_data.get(field, ""))  # Naplnění dat z `selected_data`
                entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = entry

        # Nastavení roztažení sloupců v main_frame
        main_frame.columnconfigure(1, weight=1)

        # Rámec pro tlačítka
        button_frame = ttk.Frame(popup, padding=10, style="Dark.TFrame")
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Tlačítka pro uložení nebo zrušení
        ttk.Button(button_frame, text="Save", style="Save.TButton", command=lambda: self.save_changes(entries, selected_item, popup)).grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Cancel", style="Cancel.TButton", command=popup.destroy).grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Nastavení roztažení sloupců v button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

    def save_changes(self, entries, selected_item, popup):
        # Uložení změn
        new_values = [self.tree.item(selected_item, "values")[0]] + [entries[field].get() for field in entries]
        self.tree.item(selected_item, values=new_values)
        print("Device has been updated:", new_values)
        popup.destroy()

    def addNew(self):
        # Vytvoření popup okna
        popup = tk.Toplevel(self)
        popup.title("Add New Device")
        popup.geometry("400x400")
        popup.configure(bg="#333333")  # Tmavé pozadí popup okna
        popup.resizable(False, False)

        # Nastavení gridu popup okna
        popup.rowconfigure(0, weight=1)  # Horní prostor
        popup.rowconfigure(1, weight=1)  # Prostor pro main_frame
        popup.rowconfigure(2, weight=1)  # Spodní prostor
        popup.columnconfigure(0, weight=1)  # Vycentrování ve sloupci

        # Definice stylu pro hlavní rámec
        style = ttk.Style()
        style.configure("Light.TFrame", background="#999999")  # Světlejší pozadí
        style.configure("Dark.TFrame", background="#333333")  # Tmavší pozadí
        style.configure("Popup.TLabel", background="#333333", foreground="white", font=("Arial", 20, "bold"))
        style.configure("main_frame.TLabel", background="#999999", foreground="black", font=("Arial", 9, "bold"))

        # Nadpis
        title_label = ttk.Label(popup, text="Add New Device", style="Popup.TLabel", anchor="center")
        title_label.grid(row=0, column=0, pady=(5, 5), padx=5, sticky="nsew")


        # Hlavní rámec
        main_frame = ttk.Frame(popup, padding=(5, 5), style="Light.TFrame")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        # Pole pro zadání dat
        fields = ["Type", "Manufacturer", "Serial Number", "Status", "Location"]
        combobox_data = {
            "Type": Type,
            "Manufacturer": Manufacturer,
            "Status": Status,
            "Location": Location
        }
        entries = {}
        for i, field in enumerate(fields_frame1):
            ttk.Label(main_frame, text=field, style="main_frame.TLabel").grid(row=i, column=0, padx=10, pady=5, sticky="w")
            if field in combobox_data:
                # Použití Comboboxu pro pole s předdefinovanými hodnotami
                combobox = ttk.Combobox(main_frame, values=combobox_data[field], state="readonly")
                combobox.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = combobox
            else:
                # Použití Entry pro ostatní pole
                entry = ttk.Entry(main_frame)
                entry.grid(row=i, column=1, padx=10, pady=5, sticky="ew")
                entries[field] = entry

        # Nastavení roztažení sloupců v main_frame
        main_frame.columnconfigure(1, weight=1)

        # Rámec pro tlačítka
        button_frame = ttk.Frame(popup, padding=10, style="Dark.TFrame")
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        # Tlačítka pro uložení nebo zrušení
        ttk.Button(button_frame, text="Save", command=lambda: save_data()).grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        ttk.Button(button_frame, text="Cancel", command=popup.destroy).grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        # Nastavení roztažení sloupců v button_frame
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

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

    def deleteDevice(self):
        selected_items = self.tree.selection()
        for selected_item in selected_items:
            values = self.tree.item(selected_item, "values")
            self.tree.delete(selected_item)
            print("Deleting device:", values)
        
    def change_image(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")],
            title="Select an image"
        )
        if filepath:
            try:
                image = Image.open(filepath)
                image = image.resize((300, 200))  # Změna velikosti obrázku
                photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=photo)
                self.image_label.image = photo  # Uložení reference na obrázek
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")

    def deleteDevicePopup(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "No device selected!")
            return

        # Vytvoření popup okna
        popup = tk.Toplevel(self)
        popup.title("Delete Device")
        popup.geometry("400x150")
        popup.configure(bg="#333333")

        # Zobrazení varování
        message = "Are you sure you want to delete the selected device(s)?"
        ttk.Label(popup, text=message).pack(padx=10, pady=10)

        # Funkce pro potvrzení smazání
        def confirm_delete():
            self.deleteDevice()
            popup.destroy()

        # Tlačítka pro potvrzení nebo zrušení
        ttk.Button(popup, text="Yes", command=confirm_delete).pack(side="left", padx=10, pady=10)
        ttk.Button(popup, text="No", command=popup.destroy).pack(side="right", padx=10, pady=10)


if __name__ == "__main__":
    app = InventoryApp()
    app.mainloop()
