from tkinter import *
from tkinter import ttk


class myApp:
    def __init__(self, master):
        self.bu = ttk.Button(master, text='Konec', command=root.destroy)
        self.bu.pack()
        button1 = ttk.Button(root, text="tlaitko 1", command=self.log_click("hello"))
        button1.pack(side='bottom', expand='true', fill='both')

        textbox1 = ttk.Entry(root)
        textbox1.pack()

    def log_click(self, message):
        print(message)



root = Tk()
app = myApp(root)
root.geometry('800x600')
root.title("My app")
root.mainloop()