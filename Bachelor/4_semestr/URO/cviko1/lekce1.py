from tkinter import *

class myApp:
  def __init__(self, master):
    self.fr = Frame(master)
    self.input = Label(self.fr, text="Input")
    self.output = Label(self.fr, text="Output")
    self.input.pack()
    self.output.pack()
    self.convert = Button(self.fr, text="Convert", command=self.fr.quit)
    self.fr.pack()
    self.convert.pack()

root = Tk()
app = myApp(root)
root.mainloop()
