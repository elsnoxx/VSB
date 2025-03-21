from tkinter import *
import tkinter.font

class myApp:
    def update_color(self, event=None):
        r = self.r_slider.get()
        g = self.g_slider.get()
        b = self.b_slider.get()
        hex_color = f'#{r:02x}{g:02x}{b:02x}'
        self.color_display.config(bg=hex_color)
        self.hex_label.config(text=f'HEX: {hex_color}')
    
    def prevod_teploty(self, event=None):
        try:
            v = float(self.ent_in.get())
            if self.dir.get() == 1:
                result = v * 9/5 + 32  # C -> F
                unit = "F"
            else:
                result = (v - 32) * 5/9  # F -> C
                unit = "C"
            
            self.ent_out.config(state='normal')
            self.ent_out.delete(0, END)
            if unit == "C":
                self.ent_out.insert(0, f"{round(result, 2)} °{unit}")
            else:
                self.ent_out.insert(0, f"{round(result, 2)} {unit}")
            self.ent_out.config(state='readonly')
        except ValueError:
            self.ent_out.config(state='normal')
            self.ent_out.delete(0, END)
            self.ent_out.insert(0, "Chyba")
            self.ent_out.config(state='readonly')
            
    
    def show_about(self):
        about_window = Toplevel()
        about_window.title("O aplikaci")
        Label(about_window, text="Převodník jednotek", font=("Arial", 14)).pack(pady=10)
        Label(about_window, text="Verze 1.0").pack()
        Button(about_window, text="Zavřít", command=about_window.destroy).pack(pady=10)
    
    def show_rgb_converter(self):
        self.clear_frames()
        self.r_slider.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.g_slider.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        self.b_slider.grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        self.hex_label.grid(row=3, column=0, padx=10, pady=10, sticky="ew")
        self.color_display.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    def show_temp_converter(self):
        self.clear_frames()
        self.radio_frame.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.ent_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.lbl_in.grid(row=0, column=0, padx=5, pady=5)
        self.ent_in.grid(row=0, column=1, padx=5, pady=5)
        self.lbl_out.grid(row=1, column=0, padx=5, pady=5)
        self.ent_out.grid(row=1, column=1, padx=5, pady=5)
        self.btn_convert.grid(row=2, column=0, columnspan=2, pady=5)
        self.ca.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
    
    def clear_frames(self):
        for widget in self.left_frame.winfo_children():
            widget.grid_forget()
        for widget in self.right_frame.winfo_children():
            widget.grid_forget()
    
    def change_image(self):
        try:
            self.photo = PhotoImage(file="th.png")
            self.ca.create_image(150, 200, image=self.photo)
        except Exception as e:
            print("Chyba načtení obrázku:", e)
    
    def __init__(self, root):
        root.title('Převodník')
        root.resizable(False, False)
        root.geometry("600x450")

        def_font = tkinter.font.nametofont("TkDefaultFont")
        def_font.config(size=16)

        menu_frame = Frame(root)
        menu_frame.grid(row=0, column=0, columnspan=2, sticky="ew")

        menu_bar = Menu(menu_frame)
        root.config(menu=menu_bar)
        
        menu_options = Menu(menu_bar, tearoff=0)
        menu_options.add_command(label="Teploměr", command=self.show_temp_converter)
        menu_options.add_command(label="Převod barev", command=self.show_rgb_converter)        
        
        menu_bar.add_cascade(label="Možnosti", menu=menu_options)
        menu_bar.add_command(label="O aplikaci", command=self.show_about)
        menu_bar.add_command(label="Konec", command=root.quit)
        
        version_label = Label(menu_frame, text="FIC0024", anchor=E)
        version_label.grid(row=0, column=1, padx=10)
        
        self.left_frame = Frame(root)
        self.right_frame = Frame(root, width=300, height=400)
        
        self.r_slider = Scale(self.left_frame, from_=0, to=255, orient=HORIZONTAL, label="R", command=self.update_color)
        self.g_slider = Scale(self.left_frame, from_=0, to=255, orient=HORIZONTAL, label="G", command=self.update_color)
        self.b_slider = Scale(self.left_frame, from_=0, to=255, orient=HORIZONTAL, label="B", command=self.update_color)
        
        self.color_display = Label(self.right_frame, width=20, height=10, bg="#000000")
        self.hex_label = Label(self.left_frame, text="HEX: #000000", font=def_font)
        
        self.dir = IntVar()
        self.dir.set(1)
        
        self.radio_frame = LabelFrame(self.left_frame, text="Směr převodu")
        self.radio_c_to_f = Radiobutton(self.radio_frame, text="°C -> F", variable=self.dir, value=1)
        self.radio_f_to_c = Radiobutton(self.radio_frame, text="F -> °C", variable=self.dir, value=2)
        self.radio_c_to_f.grid(row=0, column=0, padx=5, pady=5)
        self.radio_f_to_c.grid(row=0, column=1, padx=5, pady=5)
        
        self.ent_frame = LabelFrame(self.left_frame, text="Vstup a výstup", padx=10, pady=10, bd=2, relief=GROOVE)
        self.lbl_in = Label(self.ent_frame, text="Vstup")
        self.ent_in = Entry(self.ent_frame, width=10, font=def_font)
        self.lbl_out = Label(self.ent_frame, text="Výstup")
        self.ent_out = Entry(self.ent_frame, width=10, font=def_font, state='readonly')
        self.btn_convert = Button(self.ent_frame, text="Převést", command=self.prevod_teploty)
        
        self.ca = Canvas(self.right_frame, width=300, height=500)
        try:
            self.photo_empty = PhotoImage(file="th_empty.png")
            self.ca.create_image(150, 200, image=self.photo_empty)
        except Exception as e:
            print("Chyba načtení obrázku:", e)
        
        self.left_frame.grid(row=1, column=0, sticky="nsew")
        self.right_frame.grid(row=1, column=1, sticky="nsew")
        
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(1, weight=1)
        
        self.show_temp_converter()

root = Tk()
app = myApp(root)
root.mainloop()