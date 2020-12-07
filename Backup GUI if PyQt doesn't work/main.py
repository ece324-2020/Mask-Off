import PIL
from PIL import Image, ImageTk
import cv2
import tkinter as tk

LARGE_FONT = ("Verdana", 12)


class MaskOff(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.container = tk.Frame(self)

        self.container.pack(side="top", fill="both", expand=True)

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        width, height = 800, 600
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frames = {}
        self.active_frame = Welcome

        frame = Welcome(self.container, self)
        self.frames[Welcome] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(Welcome)

    def remove_frame(self, cont):
        self.frames[cont].grid_forget()
        self.frames[cont].destroy()
        del self.frames[cont]

    def show_frame(self, cont, model=None):
        if cont not in self.frames:
            frame = cont(self.container, self, model)
            self.frames[cont] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        frame = self.frames[cont]
        self.active_frame = cont
        frame.tkraise()


class Welcome(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self,
                         text='Weclome to "Mask Off" - a tool designed to classify how well a mask is being worn.',
                         font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button = tk.Button(self, text="See how well you are personally wearing a mask",
                           command=lambda: controller.show_frame(Webcam, "one person"))
        button.pack()

        button2 = tk.Button(self, text="See how the public is doing",
                            command=lambda: controller.show_frame(Webcam, "multiple people"))
        button2.pack()


class Webcam(tk.Frame):
    def __init__(self, parent, controller, model):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.model = model

        label = tk.Label(self, text="Please show your face blah blah", font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        self.lmain = tk.Label(self)
        self.lmain.pack()

        button1 = tk.Button(self, text="Back to Home", command=self.exit)
        button1.pack()

        self.show_frame()

    def exit(self):
        self.controller.show_frame(Welcome)
        self.controller.remove_frame(Webcam)

    def show_frame(self):
        print(self.model)
        _, frame = self.controller.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

        """
            This is where we would apply `self.model`. cv2image is a numpy array of the feed from the webcam.
            Need to create a function to resize it and apply data cleaning etc.
        """

        img = PIL.Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.imgtk = imgtk
        self.lmain.configure(image=imgtk)
        self.lmain.after(10, self.show_frame)


app = MaskOff()
app.mainloop()
