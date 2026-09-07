# PyCharm & Python Veri Bilimi Çalışma Ortamı Günlüğü ve Sorun Giderme Kılavuzu

Bu dosya, **miuul2** projesinde karşılaşılan yavaşlık, donma ve veri tipi kaynaklı hataların çözümlerini ve gelecekte benzer bir problemle karşılaştığınızda uygulayabileceğiniz adımları içerir.

---

## 🛠️ Karşılaşılan Sorunlar ve Yapılan Düzenlemeler

### 1. PyCharm'ın Açılışta ve Kod Yazarken Aşırı Kasması (İndeksleme Yükü)
* **Sorun:** Proje açılır açılmaz bilgisayarın kilitlenmesi ve disk/işlemci kullanımının tavan yapması.
* **Sebep:** PyCharm'ın varsayılan olarak `dataset/` klasöründeki **158 MB** boyutundaki `application_train.csv` dosyasını kelime kelime indekslemeye çalışması.
* **Çözüm:** `miuul2` projesinin `.idea/miuul2.iml` dosyasına dışlama kuralı eklendi. `dataset/` klasörü **"Excluded" (Dışlanmış)** olarak işaretlendi.
  * *Not:* Bu ayar Pandas'ın dosyayı okumasını **engellemez**, sadece PyCharm'ın arka planda dosyayı boşuna taramasını engeller.

### 2. Yan Ekranda `feature_engineering.py` Açıkken Yaşanan Donmalar
* **Sorun:** Editörde iki dosya açıkken ve aktif olarak `FE1.py` yazılırken sürekli donmalar yaşanması.
* **Sebep:** `FE1.py` dosyasının en üstünde `from Downloaded.feature_engineering import cat_but_car` satırının bulunması. Python'da import işlemi o dosyanın tamamen çalıştırılmasına yol açar. Referans olarak baktığınız 1113 satırlık `feature_engineering.py` dosyası ise arka planda 158 MB'lık dosyayı 3-4 kez yüklüyor, makine öğrenimi modelleri eğitiyor ve `plt.show()` grafikleri açarak ekranı kilitliyordu.
* **Çözüm:** Bu gereksiz import satırı [FE1.py](file:///C:/Users/infka/Desktop/miuul2/Fengineering/FE1.py) dosyasından tamamen temizlendi.

### 3. Redundant (Gereksiz) Veri Yükleme Yavaşlığı
* **Sorun:** Kodun çalıştırılmasının saniyeler sürmesi.
* **Sebep:** `FE1.py` dosyasında 29. satırda 158 MB'lık `application_train.csv` dosyası `df` değişkenine yükleniyor, ancak hemen ardından 38. satırda Titanic verisi ile hiç kullanılmadan eziliyordu.
* **Çözüm:** Bu satır yorum satırına alınarak her çalıştırmada 5-10 saniye zaman kazanıldı.

### 4. `TypeError: unsupported operand type(s) for -: 'str' and 'str'` Hatası
* **Sorun:** `for col in num_cols: print(col, check_outlier(df, col))` döngüsünü çalıştırırken numpy quantile hesaplamasında hata alınması.
* **Sebep:** Kullandığınız Python 3.14 ve modern Pandas sürümünde metin kolonları (`Name`, `Sex`, `Ticket` vb.) artık klasik `"O"` (object) tipi yerine doğrudan native `"str"` veya `"string"` veri tipiyle yüklenmektedir. Orijinal `grab_col_names` fonksiyonunda sadece `"O"` kontrolü yapıldığı için metin kolonları sayısal (`num_cols`) sanılıp matematiksel çeyreklik (quantile) işlemine sokuluyordu.
* **Çözüm:** `grab_col_names` fonksiyonunun tip kontrol mekanizması modern Python standartlarına göre şu şekilde güncellendi:
  ```python
  dataframe[col].dtype.name in ["object", "category", "string", "str"]
  ```

---

## 🚀 Gelecekte PyCharm Yine Kasarsa veya Hata Verirse Ne Yapılmalı?

Eğer ileride PyCharm'da garip donmalar veya kod çalışırken tutarsızlıklar yaşarsanız şu **4 adımlı sorun giderme protokolünü** uygulayın:

### 1. Önbellekleri Temizleme (Invalidate Caches)
PyCharm bazen eski indeksleri hafızasında yanlış tutar ve kısırdöngüye girer.
* Üst menüden **File -> Invalidate Caches...** seçeneğine tıklayın.
* Açılan penceredeki **tüm kutucukları işaretleyin** ve **Invalidate and Restart** deyin.
* Bu işlem tüm geçici hafızayı siler ve dışlama kurallarını sıfırdan temizce çalıştırır.

### 2. PyCharm Bellek Sınırını (Heap Size) Kontrol Edin
Veri bilimi yaparken PyCharm'a ayrılan RAM yetersiz kalıp Garbage Collection (çöp toplama) kilitlemelerine sebep olabilir.
* Üst menüden **Help -> Change Memory Settings** seçeneğine gidin.
* Limit değerini en az **4096 MiB (4 GB)** veya daha fazlasına yükseltip kaydedin.

### 3. Konsolu (Python Console) Sıfırlama
`Alt + Shift + E` ile çalışırken hafızadaki değişkenler veya fonksiyonlar eski kalıp kodunuzu bozabilir.
* PyCharm altındaki **Python Console** sekmesinde sol tarafta yer alan yeşil dairesel ok (Rerun) butonuna basın.
* Bu buton konsolunuzun hafızasını tamamen sıfırlar. Ardından kodlarınızı yukarıdan aşağıya sırayla tekrar konsola gönderin.

### 4. Kod Hücreleri (`# %%`) ile Çalışın
Kod seçerken fareyle eksik satır seçme hatasını önlemek için kod bloklarınızın arasına `# %%` ekleyerek hücreler oluşturun ve bu hücrelerin içine tıklayıp **Ctrl + Enter** kısayolu ile tek seferde çalıştırın.
