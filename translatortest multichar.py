import tkinter
import cv2
import win32gui
import PIL.Image, PIL.ImageTk
from tkinter import messagebox

import imutils
import numpy as np
from sklearn.metrics import pairwise
import pickle
import os
import time
import tensorflow as tf
from tkinter import *
from imutils.contours import sort_contours
import imutils
import numpy as np
import tensorflow.keras
from PIL import Image, ImageOps,ImageTk
import numpy as np
import os
import PIL
from PIL import Image, ImageDraw , ImageGrab

import tensorflow.keras
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
        self.key={'a':'Stop','b':'Hello','k':'Peace','l':'Love','f':'Okay','r':'Fingers Crossed','o':'Okay','c':'Okay','n':'Okay'}
        self.delay = 15

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.ui()
    def ui(self):

         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width = self.vid.width, height = self.vid.height,borderwidth=0,highlightthickness=0)
        self.canvas.pack()



         # Button that lets the user take a snapshot

        self.btn_predict=tkinter.Button(self.window, text="Predict", width=50, command=self.snapshot,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_predict.pack(anchor=tkinter.CENTER, expand=True)
        self.label=tkinter.Label(self.window,textvariable=self.var,font=("Arial", 25),bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.label.pack()
        self.M=Menu(self.window)

        self.window.config(menu=self.M,bg='black')
        self.j = Menu(self.M)

        self.j.add_command(label='Create Database', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)

        #self.pack(fill=BOTH, expand=1)


         # After it is called once, the update method will be automatically called every delay milliseconds4

        self.update()
        self.window.iconbitmap('C:\\Users\\EZIO\\Downloads\\gesture-recognition-master\\translator minor\\icon.ico')
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
            if chr(fingers) in self.key:
                self.var.set('Prediction:'+self.key[chr(fingers)])
            else:

                self.var.set('Prediction:'+chr(fingers))



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



    # global variables


    # --------------------------------------------------
    # To find the running average over the background
    # --------------------------------------------------
    def run_avg(self,image):
                # initialize the background
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return

        # compute weighted average, accumulate it and update the background
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)

    # ---------------------------------------------
    # To segment the region of hand in the image
    # ---------------------------------------------
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

    # --------------------------------------------------------------

    # --------------------------------------------------------------
    def predict(self,x):
        cv2.imwrite('file.jpg', x)

        # Disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model
        # The 'length' or number of images you can put into the array is
        # determined by the first position in the shape tuple, in this case 1.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        # image = Image.open('hey.jpg')

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)

        # turn the image into a numpy array
        # image_array = np.asarray(image)
        image = cv2.imread('file.jpg', cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)

        # display the resized image
        # image.show()

        # Normalize the image
        normalized_image_array = (image.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array
        loaded_model = tf.keras.models.load_model('gesture_model.h5')
        # run the inference
        pred = loaded_model.predict(data)


        #loaded_model = tf.keras.models.load_model('gesturemodel.h5')


        #os.remove('file.jpg')
        #os.remove('result.jpg')

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
        self.canvas = tkinter.Canvas(self.window, width = self.vid.width, height = self.vid.height,borderwidth=0,highlightthickness=0)
        self.canvas.pack()



         # Button that lets the user take a snapshot


        self.entry = tkinter.Entry(self.window)
        self.entry.pack()
        self.btn_save = tkinter.Button(self.window, text="Save", width=50, command=self.snapshot,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_save.pack(anchor=tkinter.CENTER)
        self.btn_train = tkinter.Button(self.window, text="Train", width=50, command=self.train,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_train.pack(anchor=tkinter.CENTER)
        self.M = Menu(self.window)

        self.window.config(menu=self.M,bg='black')
        self.j = Menu(self.M)

        self.j.add_command(label='Prediction', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)
        #self.pack(fill=BOTH, expand=1)


         # After it is called once, the update method will be automatically called every delay milliseconds4

        self.update()
        self.window.iconbitmap('C:\\Users\\EZIO\\Downloads\\gesture-recognition-master\\translator minor\\icon.ico')
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

        self.la = Label(self.window, text="Steps.\n 1.Create(optional) \n 2.Load \n 3.Train",bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.la.pack()

        self.label3 = Label(self.window, text='####### Create Your Own Dataset #######',bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.label3.pack()

        self.cv = Canvas(self.window, width=200, height=275, bg='white',borderwidth=0,highlightthickness=0)
        # --- PIL
        #self.image1 = PIL.Image.new('RGB', (200, 275), color='white')
        #self.draw = ImageDraw.Draw(self.image1)
        # ----
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.pack()
        self.btn_clr = Button(text="Clear Canvas", command=self.clr,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_clr.pack()
        self.la6 = Label(self.window, text='Enter Letter Drawn',bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.la6.pack()
        self.entry = Entry()
        self.entry.pack()

        self.btn_save = Button(text='Save Image', command=self.save,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_save.pack()

        self.labe = Label(self.window, text='######### Load Your Dataset #########',bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)

        self.btn_clr.pack()
        self.labe.pack()
        self.btn_load = Button(text="Train User Database", command=self.load,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_load.pack()
        self.M = Menu(self.window)

        self.window.config(menu=self.M,bg='black')
        self.j = Menu(self.M)

        self.j.add_command(label='Letter Recognition', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)
        self.window.iconbitmap('C:\\Users\\EZIO\\Downloads\\gesture-recognition-master\\translator minor\\icon.ico')
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

            # save postscipt image
            self.cv.postscript(file='file' + '.eps')
            # use PIL to convert to PNG
            img = Image.open('file' + '.eps')
            img.save(y)

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

        arr = np.zeros((y, 80, 80, 1), dtype='uint8')
        while i < y:
            filename = 'data/letter/' + str(i) + '.jpg'
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            cntrs = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            img3 = np.zeros((1, 80, 80, 1),dtype='uint8')
            for cnt in cntrs:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                top = int(0.05 * th.shape[0])
                bottom = top
                left = int(0.05 * th.shape[1])
                right = left
                # img_up=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_REPLICATE)
                roi =gray[y - top:y + h + bottom, x - left:x + w + right]
                if i == 0:
                    img5=Image.fromarray(roi)
                    img5.save('hey.jpg')
                    print(roi.shape)

                img3 = cv2.resize(roi, (80, 80), interpolation=cv2.INTER_AREA)




                img3 = img3.reshape(1, 80, 80, 1)



            arr[i] = img3


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
        x_train = x_train.reshape(x_train.shape[0], 80, 80, 1)

        input_shape = (80, 80, 1)
            # Making sure that the values are float so that we can get decimal points after division

        x_train = x_train.astype('float32')

            # Normalizing the RGB codes by dividing it to the max RGB value.
        #x_train = x_train / 255

            # Importing the required Keras modules containing model and layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
            # Creating a Sequential Model and adding the layers
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())  # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(26, activation=tf.nn.softmax))
        model.compile(optimizer='adam',

                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        model.fit(x=x_train, y=y_train, epochs=100, batch_size=16)

        try:
            os.remove('lettermodel.h5')
            model.save('lettermodel.h5')
        except:
            model.save('lettermodel.h5')
        #except:
            #messagebox.showerror("Error", "Empty Database")

    def clr(self):
        self.cv.delete('all')
        #self.image1.paste('white', (0, 0, 200, 275))

    def paint(self,event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval((x1, y1, x2, y2), fill='black', width=10)
        #  --- PIL
        #self.draw.line((x1, y1, x2, y2), fill='black', width=10)


class letterrecog:
    def __init__(self , window, window_title):
        self.window=window
        self.window.resizable(0,0)
        self.window.title(window_title)

        self.window.geometry('600x400')
        self.cv = Canvas(self.window, width=500, height=275, bg='white',borderwidth=0,highlightthickness=5,highlightbackground="black")
        # --- PIL
        #self.image1 = PIL.Image.new('RGB', (200, 275), color='white')
        #self.draw = ImageDraw.Draw(self.image1)
        # ----
        self.cv.bind('<B1-Motion>', self.paint)
        self.cv.pack()

        self.btn_save = Button(text="Predict", command=self.predict,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_save.pack()
        self.btn_clr = Button(text="Clear", command=self.clr,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.btn_clr.pack()
        self.var = StringVar()
        self.label = Label(self.window, textvariable=self.var,font=("Arial", 25),bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
        self.label.pack()
        self.M = Menu(self.window)

        self.window.config(menu=self.M,bg='black')
        self.j = Menu(self.M)

        self.j.add_command(label='Handwritten Letter Database', command=self.data)
        self.j.add_command(label='Translator Home', command=self.home)
        self.j.add_command(label='EXIT', command=self.quit)
        self.M.add_cascade(label='FILE', menu=self.j)
        self.window.iconbitmap('C:\\Users\\EZIO\\Downloads\\gesture-recognition-master\\translator minor\\icon.ico')
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
        filename = 'file'

        #save postscipt image
        self.cv.postscript(file=filename + '.eps')
        # use PIL to convert to PNG
        img = Image.open(filename + '.eps')
        img.save(filename + '.jpg')
        
        img = cv2.imread('file.jpg',cv2.IMREAD_COLOR)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,th=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


        loaded_model = tf.keras.models.load_model('letter_model.h5')
        cntrs=cv2.findContours(th,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        #cntrs = imutils.grab_contours(cntrs)
        cntrs=sort_contours(cntrs, method="left-to-right")[0]

        st=''
        print(len(cntrs))
        for cnt in cntrs:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),1)
            top=int(0.05*th.shape[0])
            bottom=top
            left=int(0.05*th.shape[1])
            right=left
            #img_up=cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_REPLICATE)
            roi=gray[y-top:y+h+bottom,x-left:x+w+right]

            img3=cv2.resize(roi,(80,80),interpolation=cv2.INTER_AREA)
            img5 = Image.fromarray(img3)
            img5.save('hey.jpg')







            # Disable scientific notation for clarity
            np.set_printoptions(suppress=True)



            # Create the array of the right shape to feed into the keras model
            # The 'length' or number of images you can put into the array is
            # determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

            # Replace this with the path to your image
            #image = Image.open('hey.jpg')

            # resize the image to a 224x224 with the same strategy as in TM2:
            # resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)

            # turn the image into a numpy array
            #image_array = np.asarray(image)
            image = cv2.imread('hey.jpg', cv2.IMREAD_COLOR)
            image = cv2.resize(image, size)

            # display the resized image
            #image.show()

            # Normalize the image
            normalized_image_array = (image.astype(np.float32) / 127.0) - 1

            # Load the image into the array
            data[0] = normalized_image_array

            # run the inference
            pred = loaded_model.predict(data)


            #pred = loaded_model.predict([img3])[0]
            st+=chr(97+pred.argmax())






        self.var.set('Prediction :' + st)

        #os.remove('file.jpg')
        #os.remove('result.jpg')

    def clr(self):
        self.cv.delete('all')
        #self.image1.paste('white', (0, 0, 200, 275))
        self.var.set('Prediction :')

    def paint(self,event):
        x1, y1 = (event.x), (event.y)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.cv.create_oval((x1, y1, x2, y2),fill='black', width=10)
        #  --- PIL
        #self.draw.line((x1, y1, x2, y2), fill='black', width=10)


def Data():
    Database(tkinter.Tk(), 'Database Generator')

def gesture():
    App(tkinter.Tk(), "Gesture Recognition")

def letmodel():
    lettermodel(tkinter.Tk(), 'Text Recognition Model')
def letrecog():
    letterrecog(tkinter.Tk(), 'Text Recognition')

def main():
    def letrecogmain():
        window.destroy()
        letrecog()
    def gesturemain():
        window.destroy()
        gesture()
    window=tkinter.Tk()
    window.title('Translator')
    window.geometry('300x300')

    path = "icon.jpeg"

    img = ImageTk.PhotoImage(Image.open(path))

    # The Label widget is a standard Tkinter widget used to display a text or image on the screen.
    panel = tkinter.Label(window, image=img,borderwidth=0,highlightthickness=0)

    # The Pack geometry manager packs widgets in rows or columns.
    panel.pack(side="top")
    btn_letter = Button(window, text='Text Recognition', command=letrecogmain, width=20,
          height=1,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
    btn_letter.pack()
    btn_gesture = Button(window, text='Gesture Recognition', command=gesturemain, width=20,
          height=1,bg='black', fg='White',relief='raised',borderwidth=0,highlightthickness=0)
    btn_gesture.pack()

    window.configure(bg='black')
    window.iconbitmap('C:\\Users\\EZIO\\Downloads\\gesture-recognition-master\\translator minor\\icon.ico')
    window.mainloop()



if __name__=='__main__':
    main()

