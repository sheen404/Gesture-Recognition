import tkinter
from gtts import gTTS
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import messagebox
import playsound

import imutils
import numpy as np
from sklearn.metrics import pairwise
import pickle
import os
import time
import tensorflow as tf
from tkinter import *

import numpy as np

import os
import PIL
from PIL import Image, ImageDraw
import PIL.ImageOps
from keras.models import load_model
import cv2

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.bg = None
        self.noframes=0
        self.accumWeight = 0.5

        self.var = StringVar()

        self.delay = 15

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.ui()
    def ui(self):

         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()



         # Button that lets the user take a snapshot

        self.btn_predict=tkinter.Button(self.window, text="Predict", width=50, command=self.snapshot)
        self.btn_predict.pack(anchor=tkinter.CENTER, expand=True)
        self.label=tkinter.Label(self.window,textvariable=self.var)
        self.label.config(font=("Courier", 44))
        self.label.pack()
        self.M=Menu(self.window)

        self.window.config(menu=self.M)
        self.j = Menu(self.M)

        self.j.add_command(label='Create Database', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)

        #self.pack(fill=BOTH, expand=1)


         # After it is called once, the update method will be automatically called every delay milliseconds4

        self.update()

        self.window.mainloop()
    def home(self):
        self.window.destroy()
        main()
    def save(self):
        pass
    def quit(self):
        self.window.destroy()
    def data(self):
        self.window.destroy()
        Data()
    def snapshot(self):

         # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.flip(frame, 1)
            h = frame.copy()
            roi = h[70:250, 375:580]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)
            hand = self.segment(gray)
            (thresholded, segmented) = hand
            fingers = 97 + self.predict(thresholded)
            self.var.set(chr(fingers))
            text=chr(fingers)
            obj = gTTS(text=text, lang='en', slow=False)
            obj.save("welcome.mp3")
            playsound.playsound(r'welcome.mp3', True)
            os.remove('welcome.mp3')



    def update(self):
         # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            frame = cv2.flip(frame, 1)
            if self.noframes<30:


                h=frame.copy()
                roi = h[70:250, 375:580]


                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                self.run_avg(gray)
            elif self.noframes==30:
                print('ready')
            cv2.rectangle(frame, (580, 70), (375, 250), (0, 255, 0), 2)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.noframes+=1
        self.window.after(self.delay, self.update)



    def run_avg(self,image):
                # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    def segment(self,image, threshold=25):

        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)

    def predict(self,x):
        cv2.imwrite('file.jpg', x)
        img = Image.open('file.jpg')
        img3 = PIL.ImageOps.invert(img)
        img3.save('result.jpg')
        img5 = cv2.imread('result.jpg', 0)

        img5 = cv2.resize(img5, (50, 50)).astype(np.float32)
        img5 = np.expand_dims(img5, axis=0)
        img5 = np.expand_dims(img5, axis=3)

        loaded_model = tf.keras.models.load_model('gesturemodel.h5')
        pred = loaded_model.predict(img5.reshape(1, 50, 50, 1))

        os.remove('file.jpg')
        os.remove('result.jpg')

        return (pred.argmax())


class MyVideoCapture:
    def __init__(self, video_source=0):
         # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

         # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

 # Create a window and pass it to the Application object
class Database:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.bg = None
        self.noframes=0
        self.accumWeight = 0.5

        self.var = StringVar()
        self.var.set('hey')
        self.delay = 15
        self.count=0

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.ui()
    def ui(self):

         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()



         # Button that lets the user take a snapshot


        self.entry = tkinter.Entry(self.window)
        self.entry.pack()
        self.btn_save = tkinter.Button(self.window, text="Save", width=50, command=self.snapshot)
        self.btn_save.pack(anchor=tkinter.CENTER)
        self.btn_train = tkinter.Button(self.window, text="Train", width=50, command=self.train)
        self.btn_train.pack(anchor=tkinter.CENTER)
        self.M = Menu(self.window)

        self.window.config(menu=self.M)
        self.j = Menu(self.M)

        self.j.add_command(label='Prediction', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)
        #self.pack(fill=BOTH, expand=1)


         # After it is called once, the update method will be automatically called every delay milliseconds4

        self.update()

        self.window.mainloop()
    def home(self):
        self.window.destroy()
        main()

    def quit(self):
        self.window.destroy()
    def data(self):
        self.window.destroy()
        gesture()
    def snapshot(self):
        if len(self.entry.get())!=0:

         # Get a frame from the video source
            ret, frame = self.vid.get_frame()

            if ret:

                if self.count<100:
                    frame = cv2.flip(frame, 1)
                    h = frame.copy()
                    roi = h[70:250, 375:580]

                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (7, 7), 0)
                    hand = self.segment(gray)
                    (thresholded, segmented) = hand
                    y = 'data/gesture/' + str(len(os.listdir('data/gesture'))) + '.jpg'
                    cv2.imwrite(y, thresholded)
                    try:
                        f = open('gesturedatainfo.dat', 'rb')
                        l = pickle.load(f)
                        f.close()
                        l.append(ord(self.entry.get()) - 97)
                        os.remove('gesturedatainfo.dat')
                    except:
                        l = [ord(self.entry.get()) - 97]
                    f = open('gesturedatainfo.dat', 'wb')
                    pickle.dump(l, f)
                    f.close()
                    self.count+=1
        else:
            messagebox.showerror("Error", "Empty Label")



    def update(self):
         # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if self.count==100:
            self.count=0

            print('done')
            self.entry.delete(0,len(self.entry.get()))
        if self.count<100 and self.count>0:
            self.snapshot()

        if ret:
            frame = cv2.flip(frame, 1)
            if self.noframes<30:


                h=frame.copy()
                roi = h[70:250, 375:580]


                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                self.run_avg(gray)
            elif self.noframes==30:
                print('ready')
            cv2.rectangle(frame, (580, 70), (375, 250), (0, 255, 0), 2)
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            self.noframes+=1
        self.window.after(self.delay, self.update)







    def run_avg(self,image):
                # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)


    def segment(self,image, threshold=25):

        # find the absolute difference between background and current frame
        diff = cv2.absdiff(self.bg.astype("uint8"), image)

        # threshold the diff image so that we get the foreground
        thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

        # get the contours in the thresholded image
        (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # return None, if no contours detected
        if len(cnts) == 0:
            return
        else:
            # based on contour area, get the maximum contour which is the hand
            segmented = max(cnts, key=cv2.contourArea)
            return (thresholded, segmented)
    def train(self):

        i=0
        y=len(os.listdir('data/gesture/'))

        arr = np.zeros((y, 50, 50, 1), dtype='uint8')
        while i<y:

            filename = 'data/gesture/'+str(i)+'.jpg'

            img = Image.open(filename)
            img3 = PIL.ImageOps.invert(img)
            img3.save('result.jpg')
            img5 = cv2.imread('result.jpg', 0)

            img5 = cv2.resize(img5, (50, 50)).astype(np.float32)

            img5 = np.expand_dims(img5, axis=0)
            img5 = np.expand_dims(img5, axis=3)

            arr[i]=img5




            os.remove('result.jpg')
            i+=1

        print('loaded')
        x_train = arr
        try:
            f = open('gesturedatainfo.dat', 'rb')
            Y = pickle.load(f)
            print(Y)
            f.close()
            y_train = np.array(Y, dtype='uint8')

            # Reshaping the array to 4-dims so that it can work with the Keras API
            x_train = x_train.reshape(x_train.shape[0], 50, 50, 1)

            input_shape = (50, 50, 1)
            # Making sure that the values are float so that we can get decimal points after division

            x_train = x_train.astype('float32')

            # Normalizing the RGB codes by dividing it to the max RGB value.
            x_train = x_train / 255

            # Importing the required Keras modules containing model and layers
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
            # Creating a Sequential Model and adding the layers
            model = Sequential()
            model.add(Conv2D(50, kernel_size=(3, 3), input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
            model.add(Dense(128, activation=tf.nn.relu))
            model.add(Dropout(0.2))
            model.add(Dense(26, activation=tf.nn.softmax))
            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            model.fit(x=x_train, y=y_train, epochs=20)
            try:
                os.remove('gesturemodel.h5')
                model.save('gesturemodel.h5')
            except:
                model.save('gesturemodel.h5')
        except:
            messagebox.showerror("Error", "Empty Database")


class lettermodel:
    def __init__(self,window,window_title):
        self.window=window
        self.window.title(window_title)
        self.window.geometry('325x510')

        self.la = Label(self.window, text="Steps.\n 1.Create(optional) \n 2.Load \n 3.Train")
        self.la.pack()

        self.label3 = Label(self.window, text='####### Create Your Own Dataset #######')
        self.label3.pack()

        self.cv = Canvas(self.window, width=200, height=275, bg='white')
        # --- PIL
        self.image1 = PIL.Image.new('RGB', (200, 275), color='white')
        self.draw = ImageDraw.Draw(self.image1)
        # ----
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.pack()
        self.btn_clr = Button(text="Clear Canvas", command=self.clr)
        self.btn_clr.pack()
        self.la6 = Label(self.window, text='Enter Letter Drawn')
        self.la6.pack()
        self.entry = Entry()
        self.entry.pack()

        self.btn_save = Button(text='Save Image', command=self.save)
        self.btn_save.pack()

        self.labe = Label(self.window, text='######### Load Your Dataset #########')

        self.btn_clr.pack()
        self.labe.pack()
        self.btn_load = Button(text="Train User Database", command=self.load)
        self.btn_load.pack()
        self.M = Menu(self.window)

        self.window.config(menu=self.M)
        self.j = Menu(self.M)

        self.j.add_command(label='Letter Recognition', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)

        self.window.mainloop()
    def home(self):
        self.window.destroy()
        main()
    def quit(self):
        self.window.destroy()
    def data(self):
        self.window.destroy()
        letrecog()
    def save(self):
        if len(self.entry.get()) != 0:
            y = 'data/letter/' + str(len(os.listdir('data/letter'))) + '.jpg'
            self.image1.save(y)
            try:
                f = open('letterdatainfo.dat', 'rb')
                l = pickle.load(f)
                f.close()
                l.append(ord(self.entry.get()) - 97)
                os.remove('letterdatainfo.dat')
            except:
                l = [ord(self.entry.get()) - 97]
                print(type(l))
            f = open('letterdatainfo.dat', 'wb')
            pickle.dump(l, f)
            f.close()
            self.clr()
            self.entry.delete(0, len(self.entry.get()))
        else:
            messagebox.showerror("Error", "Entry box cannot be empty")

    def load(self):

        i = 0
        y = len(os.listdir('data/letter'))

        arr = np.zeros((y, 28, 28, 1), dtype='uint8')
        while i < y:
            filename = 'data/letter/' + str(i) + '.jpg'

            img = Image.open(filename)
            img3 = PIL.ImageOps.invert(img)
            img3.save('result.jpg')
            img5 = cv2.imread('result.jpg', 0)

            img5 = cv2.resize(img5, (28, 28)).astype(np.float32)

            img5 = np.expand_dims(img5, axis=0)
            img5 = np.expand_dims(img5, axis=3)

            arr[i] = img5

            os.remove('result.jpg')
            i += 1

        print('loaded')
        x_train = arr
        #try:
        f = open('letterdatainfo.dat', 'rb')
        Y = pickle.load(f)
        print(Y)
        f.close()
        y_train = np.array(Y, dtype='uint8')

            # Reshaping the array to 4-dims so that it can work with the Keras API
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

        input_shape = (28, 28, 1)
            # Making sure that the values are float so that we can get decimal points after division

        x_train = x_train.astype('float32')

            # Normalizing the RGB codes by dividing it to the max RGB value.
        x_train = x_train / 255

            # Importing the required Keras modules containing model and layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
            # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(26, activation=tf.nn.softmax))
        model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, epochs=50)

        try:
            os.remove('lettermodel.h5')
            model.save('lettermodel.h5')
        except:
            model.save('lettermodel.h5')
        #except:
            #messagebox.showerror("Error", "Empty Database")

    def clr(self):
        self.cv.delete('all')
        self.image1.paste('white', (0, 0, 200, 275))

    def paint(self,event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval((x1, y1, x2, y2), fill='black', width=10)
        #  --- PIL
        self.draw.line((x1, y1, x2, y2), fill='black', width=10)


class letterrecog:
    def __init__(self , window, window_title):
        self.window=window
        self.window.title(window_title)

        self.window.geometry('300x350')
        self.cv = Canvas(self.window, width=200, height=275, bg='white')
        # --- PIL
        self.image1 = PIL.Image.new('RGB', (200, 275), color='white')
        self.draw = ImageDraw.Draw(self.image1)
        # ----
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.pack()

        self.btn_save = Button(text="Predict", command=self.predict)
        self.btn_save.pack()
        self.btn_clr = Button(text="Clear", command=self.clr)
        self.btn_clr.pack()
        self.var = StringVar()
        self.label = Label(self.window, textvariable=self.var)
        self.label.config(font=("Courier", 44))
        self.label.pack()
        self.M = Menu(self.window)

        self.window.config(menu=self.M)
        self.j = Menu(self.M)

        self.j.add_command(label='Handwritten Letter Database', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)

        self.window.mainloop()

    def home(self):
        self.window.destroy()
        main()

    def quit(self):
        self.window.destroy()

    def data(self):
        self.window.destroy()
        letmodel()


    def predict(self):
        filename = 'file.jpg'
        self.image1.save(filename)
        img = Image.open('file.jpg')
        img3 = PIL.ImageOps.invert(img)
        img3.save('result.jpg')
        img5 = cv2.imread('result.jpg', 0)

        img5 = cv2.resize(img5, (28, 28)).astype(np.float32)
        img5 = np.expand_dims(img5, axis=0)
        img5 = np.expand_dims(img5, axis=3)

        loaded_model = tf.keras.models.load_model('lettermodel.h5')
        pred = loaded_model.predict(img5.reshape(1, 28, 28, 1))
        text = chr(97 + (pred.argmax()))
        self.var.set('Prediction :' + text)

        obj = gTTS(text=text, lang='en', slow=False)
        obj.save("welcome.mp3")
        playsound.playsound(r'welcome.mp3', True)
        os.remove('welcome.mp3')

        os.remove('file.jpg')
        os.remove('result.jpg')

    def clr(self):
        self.cv.delete('all')
        self.image1.paste('white', (0, 0, 200, 275))
        self.var.set('Prediction :')

    def paint(self,event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval((x1, y1, x2, y2), fill='black', width=10)
        #  --- PIL
        self.draw.line((x1, y1, x2, y2), fill='black', width=10)


def Data():
    Database(tkinter.Tk(), 'Database Generator')

def gesture():
    App(tkinter.Tk(), "Gesture Recognition")

def letmodel():
    lettermodel(tkinter.Tk(), 'Letter Recognition Model')
def letrecog():
    letterrecog(tkinter.Tk(), 'Letter Recognition')

def main():
    def letrecogmain():
        window.destroy()
        letrecog()
    def gesturemain():
        window.destroy()
        gesture()
    window=tkinter.Tk()
    window.title('Translator')
    window.geometry('300x50')
    btn_letter=Button(window,text='Letter Recognition',command=letrecogmain)
    btn_letter.pack()
    btn_gesture=Button(window,text='Gesture Recognition',command=gesturemain)
    btn_gesture.pack()
    window.mainloop()



if __name__=='__main__':
    main()

