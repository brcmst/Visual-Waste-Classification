import tensorflow


# derin öğrenme modeli için önce özellik çıkarılması daha sonra da sınıflandırma yapılır
# görüntüden özellik çıkarma havuzlama ve evrişim ile gerçekleşir
# bu işlemleri direkt olarak almamızı sağlayan VGG16 transfer öğrenme modeli kullanıldı 
conv_base = tensorflow.keras.applications.VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3)
                  )

# Convolutional layers.
#conv_base.summary()

# 'block5_conv1'e kadar olan parametrelerin eğitimlerinin durması için dondurulur, çünkü tekrar eğitmeye gerek yok
# 21 milyon parametre var hepsi eğitilebilirdi, frozen ile 13 milyonu eğitilebilir oldu
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True  
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

# boş model oluştur
model = tensorflow.keras.models.Sequential()

# modelin birinci katmanına özellik çıkaran model eklenir
model.add(conv_base)

# matrisler flatten ile vektor haline getirilir
model.add(tensorflow.keras.layers.Flatten())

# düzleştirme işleminden sonra 256 tane sınıflandırıcı nöron ekle
# 3 sınıflı model olduğu için en sona softmax ile 3 nöron ekle 
model.add(tensorflow.keras.layers.Dense(256, activation='relu'))
model.add(tensorflow.keras.layers.Dense(3, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# oluşturulan model gösterilir
#model.summary()


# egitim ve test yolu, görüntüler verilir
train_dir = 'C:/Users/asus/Desktop/AtikTanimaProjesi/arsiv/foto/training/'
test_dir = 'C:/Users/asus/Desktop/AtikTanimaProjesi/arsiv/foto/test/'
validation_dir = 'C:/Users/asus/Desktop/AtikTanimaProjesi/arsiv/foto/validation/'
# veri seti ve model oluştu...


# keras' ın önişleme modulü altında çeşitli işlemler yapılır
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator( 
      rescale=1./255, # piksel değerleri 0-255'den 0-1 arasına getiriliyor
      rotation_range=40, # istenilen veri artırma işlemleri yapılabilir
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,       # yakınlaştırma
      horizontal_flip=True, # aynalama
      fill_mode='nearest'
      )
train_generator = train_datagen.flow_from_directory(  
        train_dir,          # train dizinindeki görüntüler verilir
        target_size=(224, 224),
        batch_size=16,      # klasörden görüntüleri okurken her seferinde 16 parça okur
        )


# geçerlemede sadece sayılar 255'e bölünür
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        )

# eğitim
history = model.fit_generator(
      train_generator,       # eğitim verisi train_generator' den alınır
      steps_per_epoch=20,    
      epochs=10,
      validation_data=validation_generator,
      validation_steps=5)
      
# model kayıt etme
model.save('atik001.h5')

# eğitilen modeli test etme
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        )

# test sonuçları 
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)