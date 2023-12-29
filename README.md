‘Görüntülü Atık Sınıflandırma Projesi’ bir nesne tespiti ve tanıma projesidir. Nesne
tespiti genel olarak dijital görüntü ve videolarda belirli bir sınıfa (metal, kağıt, cam gibi) ait
olan nesnenin belirginleşmesi ve işlenecek olan nesnenin arka plandan ayrılması ile ilgilenen,
bilgisayarlı görü ve görüntü işleme ile ilgili bir bilgisayar teknolojisi olarak tanımlanmaktadır.
‘Görüntülü Atık Sınıflandırma Projesi’ 3 sınıflı görüntü sınıflandırma yapmaktadır, mevcut
olan sınıflar metal, kâğıt ve camdır.

 Resim Toplama
Bu aşamada hazırlanan veri gürültülerinden ayrıştırılmak, istenilen formata getirilmek
gibi amaçlarla sisteme girdi olarak verilir. ‘Görüntülü Atık Sınıflandırma’ için metal, cam ve
kâğıt olmak üzere 3 sınıftan oluşan veri seti bulunmaktadır. Bu veriler
- Train
- Metal (266)
- Kâğıt (286)
- Cam (291)
- Validation
- Metal (95)
- Kâğıt (95)
- Cam (97)
- Test (223)
Şeklinde veri seti oluşturuldu.

 Eğitim Verisi Oluşturma
‘Görüntülü Atık Sınıflandırma Projesi veri setinde bulunan her fotoğraf 3 sınıf için de
tek tek düzenlendi; boyutları küçültüldü, veriler en net olacak şekilde kırpıldı, farklı açılardan
kopyaları kaydedilerek veri sayısı çoğaltıldı ve sınıflardan cam için 0, metal için 1, kağıt için
ise 2 etiketleri verildi.


 Eğitim Ayarlarının Yapılması
Bu adımda projede kullanılacak model ‘Tensorflow’ kütüphanesi altında ‘Keras’ ile
geliştirilmiştir. Öncelikle derin öğrenmede görüntüden özellik çıkarabilmek adına uygulanan
işlemlerden evrişim ve havuzlama yapabilmek için ‘Tensorflow’ kütüphanesi import edilir ve
transfer öğrenme modeli olan VGG16 kullanılır böylece oluşturulan yeni modelin ilk katmanı
tamamlanır. Daha sonra 7x7’lik 512 tane matrisi tek vektör haline getirme için Flatten
(düzleştirme) uygulanır. 256 tane sınıflandırıcı nöron eklenir ve totalde 3 sınıf olduğu için
softmax ile son katmana 3 nöron eklenir.

 Eğitimin Gerçekleştirilmesi
Oluşturulan model compile edilir. Eğitim boyunca her adımda Loss değeri azalırken
Accuracy artar, sonuç olarak Accuracy için 0.90 değeri elde edilir ve model ‘atik001.h5’ adında
kaydedilir.



 Masaüstü Uygulama
Projenin bu kısmı için Python ‘Tkinter’ kütüphanesinden yararlanıldı. ‘Dosya Seçmek
İçin Tıklayınız’ butonuna basıldığında bilgisayardaki dosyalarımızı açan bir fonksiyon
tanımlandı.


Proje dosyalarının arasında bulunan test klasöründen rastgele bir metal fotoğrafı
seçilsin. Bir sonraki aşama ‘Tahmin Et’ butonuna basmaktır. Bu butona basıldığında
oluşturulan ‘atik001.h5’ modelinin kullanılarak yapılan tahmine dayalı sonuç ekrana
gelmektedir.

