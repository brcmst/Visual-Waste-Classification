from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import tensorflow.python.saved_model
import os
import numpy as np
from PIL import ImageTk, Image


#Seçilen fotoğrafın konumu daha sonra testte de kullanılacağından global değişken olarak oluşturuldu.
filepath = ""

#Fotoğraf seçiniz butonuna tıklandığında çalışacak fonk tanımlandı.
def callback():
    #Fonk içindeki filepath adlı değişkenin globaldeki filepath old belirtildi.
    global filepath
    #Açılan dosya seçiciden gelen foto konumu filepath değişkenine atandı.
    filepath  = filedialog.askopenfile().name
    #Foto konumunun gösterileceği filenamelabel adlı label elementinin text attribute ü seçilen fotonun konumu olarak ayarlandı.
    filenameLabel["text"] = filepath
    #panel["image"] = ImageTk.PhotoImage(Image.open(filepath))




#tahmin et butonuna tıklandığında çalışacak olan fonk tanımlandı.
def testModel():
    model = load_model(
        'C:/Users/asus/Desktop/AtikTanimaProjesi/atik001.h5')
    img1 = cv2.imread(filepath)
    img = cv2.resize(img1, (int(224), int(224)))
    window_name = 'image'
    cv2.imshow(window_name, img)
    img = img.reshape(1, 224, 224, 3)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    preds = model.predict(img)
    label_cam = preds[0][0]*100
    label_metal = preds[0][1]*100
    label_kagit = preds[0][2]*100
    sonucLabelCam["text"] = "Cam olma ihtimali: %"  + str(label_cam)
    sonucLabelMetal["text"] = "Metal olma ihtimali: %" + str(label_metal)
    sonucLabelKagit["text"] = "Kağıt olma ihtimali: %" + str(label_kagit)


# root değişkenine boş bir tk penceresi atıyoruz.
root = Tk()

# frm degiskenine frame in govdesini atıyoruz. Ayrıca 10 br padding veriyouz.
frm = ttk.Frame(root, padding=20)

#Çerçevemizin grid yapısını kullanmasını sağladık.
frm.grid()

filenameLabel = ttk.Label(frm, text="")
sonucLabelCam = ttk.Label(frm, text="")
sonucLabelMetal = ttk.Label(frm, text="")
sonucLabelKagit = ttk.Label(frm, text="")
ttk.Label(frm, text="Görüntü İle Atık Sınıflandırma Projesi").grid(column=0, row=0)





ttk.Button(frm, text='Dosya Seçmek İçin Tıklayınız!', command=callback).grid(column=0, row=1)

    


ttk.Label(frm, text="Seçilen Dosya: ").grid(column=0,row=2)
filenameLabel.grid(column=1,row=2)




#panel = Label(root, image = "")
#panel.grid(column=0,row=7)
ttk.Button(frm, text="Tahmin Et!", command=testModel).grid(column=0,row=3)
sonucLabelCam.grid(column=0,row=4)
sonucLabelMetal.grid(column=0,row=5)
sonucLabelKagit.grid(column=0,row=6)


ttk.Button(frm,text="Çıkış", command=root.destroy).grid(column=0, row=7)
root.mainloop()
