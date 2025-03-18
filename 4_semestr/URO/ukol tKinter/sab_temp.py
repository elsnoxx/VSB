# -*- coding: utf-8 -*-

from tkinter import *
import tkinter.font

class myApp:
    def prevod(self, event=None):
        try:
            v = float(self.ent_in.get())
            if self.dir.get() == 1:
                result = v * 9/5 + 32  # C -> F
            else:
                result = (v - 32) * 5/9  # F -> C
            
            self.ent_out.delete(0, END)
            self.ent_out.insert(0, str(round(result, 2)))
        except ValueError:
            self.ent_out.delete(0, END)
            self.ent_out.insert(0, "Chyba")

    def show_about(self):
        about_window = Toplevel()
        about_window.title("O aplikaci")
        Label(about_window, text="Převodník jednotek", font=("Arial", 14)).pack(pady=10)
        Label(about_window, text="Verze 1.0").pack()
        Button(about_window, text="Zavřít", command=about_window.destroy).pack(pady=10)
    
    def __init__(self, root):
        root.title('Převodník teplot')
        root.resizable(False, False)
        root.bind('<Return>', self.prevod)        

        def_font = tkinter.font.nametofont("TkDefaultFont")
        def_font.config(size=16)

        menu_bar = Menu(root)
        root.config(menu=menu_bar)

        menu_bar.add_command(label="O aplikaci", command=self.show_about)
        menu_bar.add_command(label="RGB")
        menu_bar.add_command(label="Konec", command=root.quit)

        self.left_frame = Frame(root, padx=10, pady=10)
        self.right_frame = Frame(root)
        
        self.dir = IntVar()
        self.dir.set(1) 
        
        self.radio_frame = LabelFrame(self.left_frame, text="Směr převodu")
        self.radio_c_to_f = Radiobutton(self.radio_frame, text="C -> F", variable=self.dir, value=1)
        self.radio_f_to_c = Radiobutton(self.radio_frame, text="F -> C", variable=self.dir, value=2)
        self.radio_c_to_f.pack(anchor="w")
        self.radio_f_to_c.pack(anchor="w")
        
        self.ent_frame = Frame(self.left_frame)
        self.lbl_in = Label(self.ent_frame, text="Input")
        self.ent_in = Entry(self.ent_frame, width=10, font=def_font)
        self.ent_in.insert(0, '0')
        self.lbl_out = Label(self.ent_frame, text="Output")
        self.ent_out = Entry(self.ent_frame, width=10, font=def_font)
        self.btn_convert = Button(self.ent_frame, text="Convert", command=self.prevod)
        
        self.ca = Canvas(self.right_frame, width=300, height=400)
        try:
            self.photo = PhotoImage(file="th.png")
            self.ca.create_image(150, 200, image=self.photo)
        except Exception as e:
            print("Chyba načtení obrázku:", e)

        self.left_frame.pack(side="left", fill=Y)
        self.right_frame.pack(side="right")
        
        self.radio_frame.pack(pady=5, fill=X)
        self.ent_frame.pack(pady=10)
        self.lbl_in.pack()
        self.ent_in.pack()
        self.lbl_out.pack()
        self.ent_out.pack()
        self.btn_convert.pack(pady=5)
        
        self.ca.pack()
        
        self.ent_in.focus_force()

root = Tk()
app = myApp(root)
root.mainloop()
