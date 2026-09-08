# %%
######################################################################
# CART
######################################################################
# Referans : machine_learning_miuul/cart.py
# Veri     : datasets/diabetes.csv
# Calistirma: proje kokunden, hucre hucre (Ctrl+Enter)
#
# !! BU DOSYADA DIKKAT:
#    - skompiler Python 3.14'te BOZUK -> ref:272-276 atla, export_text (ref:261) yeterli
#    - sistem Graphviz (dot.exe) YOK -> tree_graph() (ref:246) yerine sklearn.tree.plot_tree
######################################################################
import random
# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling using CART
# 4. Hyperparameter Optimization with GridSearchCV
# 5. Final Model
# 6. Feature Importance
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
# 8. Visualizing the Decision Tree
# 9. Extracting Decision Rules
# 10. Extracting Python/SQL/Excel Codes of Decision Rules
# 11. Prediction using Python Codes
# 12. Saving and Loading Model








# %%
# --- IMPORT'LAR ---


# pip install pydotplus
# pip install skompiler
# pip install astor
# pip install joblib


import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

from machine_learning_miuul.cart import cart_model, X_test, y_train, y_test, cv_results, cart_params, cart_best_grid, \
    tree_rules
from machine_learning_miuul.linear_regression import X_train

pd.set_option('display.max_columns',None)
pd.set_option('display.widt',500)

warnings.simplefilter(action='ignore',category=Warning)











# %%
# --- VERI YUKLEME ---
df=pd.read_csv('machine_learning_miuul/datasets/diabetes.csv')

# %%
################################################
# Decision Tree Classification: CART
################################################
# [ref: cart.py:2]

y = df['Outcome']
X = df.drop(['Outcome'],axis=1)

cart_model = DecisionTreeClassifier(random_state=1).fit(X,y)

#Confusion Matrix için y_pred
y_pred = cart_model.predict(X)

#Auc için y_prob
y_prob = cart_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))   #Her şey 1 çıkıyor, model overfit olmuş
roc_auc_score(y, y_prob) #1













# %%
##############################
# Holdout Yöntemi ile Başarı Değerlendirme
#HOLD, OUT
##############################
# [ref: cart.py:74]


X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.3,
                                                     random_state=17)



cart_model = DecisionTreeClassifier(random_state=17).fit(X_train,y_train)


#Şimdi holdout yönteminde eğitim verisi ile eğitilmiş modeli test verisi üzerinden  test  edip modeli validate edelim

#Önce bi train hatasına bakalım, overfit olduğu için model , yine 1 çkmasını bekleriz

y_pred = cart_model.predict(X_train)
y_prob = cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred))                # 1 çıkıyor
'''  precision    recall  f1-score   support
           0       1.00      1.00      1.00       350
           1       1.00      1.00      1.00       187'''
roc_auc_score(y_train,y_prob)

#Test seti üzerinden validate edelim şimdi de

y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred)) #Sonuçlar kötü çktı, overfit kanıtlandı
roc_auc_score(y_test,y_prob) #0.6739506172839506



#CV ile Hatanın kontrolünün sağlamasını yapalım, gerçekten model ne kadar kötü validationu


cart_model = DecisionTreeClassifier(random_state=17).fit(X,y)

cv_results = cross_validate(cart_model,
                            X,y,
                            cv = 5,
                            scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()


#MODELİN KÖTÜ OLDUĞUNU, OVERFİT ETTİĞİNİ KANITLADIK

#BU SORUNU ÇÖZMEMİZ GEREKİYOR ŞİMDİ







# %%
################################################
# 4. Hyperparameter Optimization with GridSearchCV
################################################
# [ref: cart.py:113]




cart_model.get_params()
#HİPERPARAMETRELER İLE İLGİLİ BİLİNMESİ GEREKENLER
'''
┌───────────────────┬──────────┬───────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
│     Parametre     │ Şu anki  │               Ne işe yarar                │                               Neden overfit'e sebep olur                               │
│                   │  değer   │                                           │                                                                                        │
├───────────────────┼──────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│                   │          │ Ağacın kaç seviye derinleşebileceğini     │ None = "istediğin kadar derinleş" demek. Ağaç her yaprağı saf (tek sınıf) oluncaya     │
│ max_depth         │ None     │ sınırlar                                  │ kadar dallanır → eğitim verisini ezberler. Hoca tam da bunu range(1, 11) ile tarıyor   │
│                   │          │                                           │ (satır 118).                                                                           │
├───────────────────┼──────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│                   │          │ Bir düğümün bölünebilmesi için içinde en  │ 2 = neredeyse her düğüm bölünebilir, tek örnekli dallara kadar iner. Yükseltirsen      │
│ min_samples_split │ 2        │ az kaç örnek olması gerektiği             │ (örn. 20) ağaç erken durur, genelleşir. Hoca bunu da tarıyor (range(2, 20), satır      │
│                   │          │                                           │ 232).                                                                                  │
├───────────────────┼──────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ min_samples_leaf  │ 1        │ Bir yaprak düğümde en az kaç örnek olması │ 1 = tek bir örnek bile kendi yaprağını oluşturabilir → aşırı özelleşme. Yükseltmek     │
│                   │          │  gerektiği                                │ overfit'i azaltır ama hoca bu derste bunu taramıyor, bilgin olsun diye söylüyorum.     │
├───────────────────┼──────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ max_leaf_nodes    │ None     │ Toplam yaprak sayısına üst sınır          │ None = sınırsız yaprak → sınırsız karmaşıklık.                                         │
├───────────────────┼──────────┼───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
│ max_features      │ None     │ Her bölünmede kaç özellik (kolon)         │ None = tüm özellikler her seferinde kullanılır (CART için normal, RF'de farklı anlam   │
│                   │          │ değerlendirilsin                          │ kazanır).                                                                              │
└───────────────────┴──────────┴───────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

'''


cart_params = {'max_depth': range(1,20),
               'min_samples_split': range(2,20)}
#ÇOK ÖNEMLİ BİR NOKTA
#scoring= default mu bırakılmalı yoksa nasıl karar verilmeli
'''
Hangi metriği optimize edeceğin, hangi hatanın senin problemin için daha pahalı olduğuna bağlı:

┌────────────────────────────────────────────────────────────────────┬──────────────────────┬───────────────────────────────────────────────────────────────────────┐
│                               Durum                                │   Öncelikli metrik   │                                 Neden                                 │
├────────────────────────────────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Sınıflar dengesiz (bizim diabetes datasetinde ~%65/%35)            │ Accuracy değil       │ Model her zaman "hayır, diyabet yok" deyip bile %65 accuracy          │
│                                                                    │                      │ tutturabilir — yanıltıcı                                              │
├────────────────────────────────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Yanlış negatif pahalı (hasta olduğu halde "sağlıklı" denmesi —     │ Recall ağırlıklı     │ Hastayı kaçırmak, yanlış alarm vermekten daha kötü                    │
│ tıbbi teşhis gibi)                                                 │ veya F1              │                                                                       │
├────────────────────────────────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Yanlış pozitif pahalı (spam filtresi gibi — normal maili spam'e    │ Precision ağırlıklı  │ Ters durum                                                            │
│ atmak)                                                             │                      │                                                                       │
├────────────────────────────────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Hem precision hem recall dengeli önemliyse                         │ F1                   │ İkisinin harmonik ortalaması, tek tarafa kaymayı cezalandırır         │
├────────────────────────────────────────────────────────────────────┼──────────────────────┼───────────────────────────────────────────────────────────────────────┤
│ Genel ayırt edicilik gücünü (eşikten bağımsız) ölçmek istiyorsan   │ ROC-AUC              │ Farklı eşik değerlerinde modelin sınıfları ne kadar iyi ayırdığını    │
│                                                                    │                      │ gösterir                                                              │
└────────────────────────────────────────────────────────────────────┴──────────────────────┴───────────────────────────────────────────────────────────────────────┘

Bizim diabetes örneğinde: dataset dengesiz olduğu için accuracy yanıltıcı olabilir — hoca bunun farkında ki cross_validate çağrılarında (satır 102, 148) hem accuracy hem f1 hem roc_auc'u birlikte raporluyor. Ama GridSearchCV'nin arama kriterini hâlâ default (accuracy) bırakmış — bu tutarsızlık, sen scoring="f1" ya da scoring="roc_auc" diye elle vererek düzeltebilirsin, GridSearchCV'ye "en iyi kombinasyonu accuracy'e göre değil, F1'e göre seç" demiş olursun.

Yani cevap: her zaman default bırakmıyoruz, problemin hangi hatayı affetmediğine göre scoring parametresini elle seçiyoruz.
'''
cart_best_grid = GridSearchCV(cart_model,
                              cart_params,
                              scoring='f1',
                              cv=5,
                              n_jobs=-1,
                              verbose=1).fit(X,y)


cart_best_grid.best_params_

cart_best_grid.best_score_

random = X.sample(1,random_state=45) #195 indexli kişi,
y.sample(1,random_state=45)

cart_best_grid.predict(random) #Dip not, final modeli kurmak zorunda değilsin, car_best_grid zaten final modeldir en iyi hpierparametreleri otomatik olarakta bulmuştur





# %%
################################################
# 5. Final Model
################################################
# [ref: cart.py:137]


cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_,random_state=17).fit(X,y)
cart_final.get_params()

#veya

cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X,y)

cv_results = cross_validate(cart_final,
                            X,y,
                            cv = 5 ,
                            scoring=['accuracy','f1','roc_auc']
                            )

cv_results['test_accuracy'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()






# %%
################################################
# 6. Feature Importance
################################################
# [ref: cart.py:158]

#Değişkeneleri sağladığı katklara göre sıralandırmamız ve öyle kullanmamız gerekmektedir.





cart_final.feature_importances_ #Bu bir attribute fonskiyon değil zaten, DecisionTreeClassifier'ın eğittikten sonra otomatik hesapladığı değerleri tutan bir attribute'tur _

#İligli fonksiyonla ilgili akla takılabilecek noktalar
'''

feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})

Bu, pandas'ın dict'ten DataFrame kurma yöntemi. Mantığı:

- pd.DataFrame()'e bir sözlük (dict) veriyorsun.
- Sözlüğün her key'i → oluşacak DataFrame'in bir kolon adı oluyor. Burada key'ler 'Value' ve 'Feature'.
- Her key'in value'su → o kolonun içeriği oluyor, ama bu değerin liste/dizi gibi bir şey olması lazım (tek bir sayı değil).

Somut örnek düşün:
model.feature_importances_   # → array([0.05, 0.60, 0.02, 0.33, ...])   (8 sayı, 8 kolon için)
features.columns             # → Index(['Pregnancies', 'Glucose', 'BloodPressure', ...])  (8 isim)

pandas bunu görünce şunu yapıyor: "iki dizi de aynı uzunlukta (8'er eleman), o zaman indekslerine göre yan yana koyayım" — yani sonuç şöyle bir tablo:

┌───────┬───────────────┐
│ Value │    Feature    │
├───────┼──────────
└───────┴───────────────┘

Yani eşleşme sıraya göre oluyor — feature_importances_'ın 0. elemanı ile columns'un 0. elemanı otomatik olarak aynı satıra düşüyor. Bunun güvenli olmasının sebebi: feature_importances_ dizisi, X'e fit edilirken kolonların DataFrame'deki sırasıyla üretiliyor, yani features.columns'un sırasıyla zaten örtüşüyor. Rastgele iki listeyi böyle birleştirmek genelde tehlikelidir (sıra kayarsa yanlış eşleşir), ama burada ikisi de aynı X'ten geldiği için garanti eşleşiyor.

Kısacası: dict'in key'i kolon adı, value'su (aynı uzunluktaki dizi) o kolonun tüm satırlarındaki değerler. Sen daha önce muhtemelen pd.read_csv(...) ile hazır tablo okuyordun, elle sıfırdan tablo kurmayı ilk kez görüyorsun — bu da pandas'ın standart, sık kullanılan bir yolu.

2. [0:num] slice'ı neden var

feature_imp.sort_values(by="Value", ascending=False)[0:num]

Önce sort_values DataFrame'i Value'ya göre büyükten küçüğe sıralıyor — yani en önemli değişken en üst satırda.

[0:num] ise bu sıralı tablodan sadece ilk num satırı alıyor — yani "en önemli ilk num değişkeni göster, gerisini gösterme" demek.

Neden gerekli: plot_importance(cart_final, X, num=5) çağrısında num=5 verdin. Diyabet datasetinde 8 kolon var. Eğer slice olmasaydı, sns.barplot 8 kolonun hepsini çizerdi — sen sadece en etkili 5 taneyi görmek istiyorsun, gerisi (etkisi az olanlar) grafiği kalabalıklaştırır ve asıl önemli olanı gözden kaçırtır.

Somut fark:
- Slice olmasaydı: 8 barlık grafik, en önemsiz 3 tanesi de dahil.
- Slice ile: sıralanmış tablonun ilk 5 satırı → sadece en önemli 5 değişkenin barı.


'''
def plot_importance(model, features, num=len(X.columns), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns}) #Anlamsız olan skorları ilgili sütunlarla eşleştirdiğimiz kısımdır.
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_final, X, num=5)

#Train ve Test hataları birlikte görselleştirilir ve ayrım noktalarından karar vermeye çalışılır?










# %%
################################################
# 7. Analyzing Model Complexity with Learning Curves (BONUS)
################################################
# [ref: cart.py:179]

#BU BÖLÜMÜN AMACI max_depth gibi bir CART hiperparametresii değiştikçe modelin eğitim verisinde mi  yoksa
#train verisinde mi daha iyi/kötü performans gösterdiğini görsel olarak takip etmek.
#Nihai amacımız overfit'in tam olarak hangi karmaşıklık seviyesinde başladığını gözle görmek.
#GridSearchCV zaten en iyi hiperparametreleri buldu, bu bölümde o hiperparametreleri inceliyoruz temelde



#validation_curve fonksyionu napıyor?
'''
- validation_curve şunu yapıyor: param_name="max_depth" dediğin hiperparametreyi, param_range=range(1,11) yani 1'den 10'a kadar her değeri deneyerek modeli defalarca eğitiyor.
- Her bir max_depth değeri için, cv=10 yani 10 farklı fold ile çapraz doğrulama yapıyor — hem train hem test skorunu her fold için kaydediyor.
- scoring="roc_auc" → performans ölçütü olarak ROC-AUC kullanılıyor.
- Dönen train_score ve test_score, 10 satır (max_depth değerleri) × 10 sütun (fold'lar) boyutunda birer matris. Yani her max_depth değeri için 10 farklı fold'un skoru var.
'''
'''
Adım adım:

1. Fonksiyonun içinde gerçekten ne oluyor:
validation_curve, her bir max_depth değeri için cv=10 fold'a göre veriyi kendi içinde bölüyor (senin X, y'yi vermenin sebebi bu — bölme işini fonksiyon yapıyor, sen train/test ayırmıyorsun). Her fold'da:
- Modeli train fold'a fit ediyor.
- O modeli hem train fold'un kendisinde hem de o fold'da ayrılmış test parçasında test ediyor.
- Yani her fold başına iki skor üretiliyor: biri "eğitildiği veriyle ne kadar iyi" (train score), biri "hiç görmediği veriyle ne kadar iyi" (test score).

2. Return sırası — burası kritik nokta:
Fonksiyon içeride bu iki skor grubunu ayrı ayrı biriktiri sabit sırayla geri döndürüyor: return train_scores,test_scores — yani her zaman önce train, sonra test. Bu, fonksiyonun tasarımı/sözleşmesi (API contract), veriden çıkarılan akıllı bir şey değil.

3. Bizim tarafımızdaki satır:
train_score, test_score = validation_curve(...)

Bu bir Python tuple unpacking işlemi. Fonksiyon iki değer içeren bir demet ((dizi1, dizi2)) döndürüyor, Python bunu pozisyona göre eşliyor: soldaki isim (train_score) demetin 1. elemanına, sağdaki isim (test_score) 2. elemanına otomatik atanıyor.

Önemli olan şu: Python burada "bu train skoru, bu test skoru" diye akıllıca bir ayrım yapmıyor — sadece sıraya bakıyor. Eğer sen yanlışlıkla
test_score, train_score = validation_curve(...)
yazsaydın, Python hiçbir hata vermezdi ama isimler ters etiketlenmiş olurdu — gerçekte train skoru olan diziye test_score derdin. Doğru sırayı bilmenin tek yolu sklearn'ün dokümantasyonu/sözleşmesi: "bu fonksiyon (train_scores, test_scores) sırasıyla döner" diye tanımlı, biz de o sıraya güvenip isimlendiriyoruz.

Yani özet: ayrım fonksiyonun iç mantığında (train fold'aluyor; bizim tarafımızdaki isimlendirme ise saf sırayagüvenerek yapılan bir eşleme, akıllı bir tanıma değil.'''
train_score, test_score = validation_curve(cart_final, X, y,
                                           param_name="max_depth",
                                           param_range=range(1, 11),
                                           scoring="roc_auc",
                                           cv=10)

#Aklıma takılan başka bir noktanın açıklaması
'''
 ikinci sorum axis=1 mantığı, anlatmışsın ama oturmadı bende

Somut bir örnekle gidelim, sayılarla daha net oturur.

Önce şekli (shape) hatırlayalım:

train_score, max_depth 1'den 10'a kadar 10 farklı değer denendiği ve her biri için cv=10 fold kullanıldığı için 10 satır × 10 sütun'luk bir matris:

- Satırlar → farklı max_depth değerleri (1, 2, 3, ..., 10)
- Sütunlar → aynı max_depth için farklı fold'ların skorları (fold 1, fold 2, ..., fold 10)

Küçük bir örnek yapalım, sadece 3 max_depth değeri ve 4 fold varmış gibi düşün (gerçekte 10x10 ama mantık aynı):

              fold1  fold2  fold3  fold4
max_depth=1 [  0.70,  0.72,  0.68,  0.71 ]
max_depth=2 [  0.80,  0.82,  0.79,  0.81 ]
max_depth=3 [  0.75,  0.60,  0.90,  0.65 ]

Senin istediğin şey ne? "Her max_depth değeri için, o satırdaki 4 fold'un ortalamasını al" — yani her satırı kendi içinde eritip tek bir sayıya
indirmek istiyorsun. Sonuçta elinde 3 sayı olmalı (her m4 değil.

axis parametresi tam bunu söylüyor: "hangi yönde eritilsin?"

- axis=1 → satır boyunca ilerle, o satırın tüm sütunlarını tek sayıya indir. Yani her satır kendi içinde ortalanır. Sonuç: 3 satır kalır (her max_depth için 1 ortalama), sütunlar yok olur.
- axis=0 → tam tersi, sütun boyunca ilerle, her sütunun tüm satırlarını tek sayıya indir. Bu durumda "her fold'un 3 max_depth'teki ortalaması" çıkardı — bizim istemediğimiz şey.

Örnek üzerinden axis=1 uygulayınca:
max_depth=1: (0.70+0.72+0.68+0.71)/4 = 0.7025
max_depth=2: (0.80+0.82+0.79+0.81)/4 = 0.805
max_depth=3: (0.75+0.60+0.90+0.65)/4 = 0.725
Sonuç: [0.7025, 0.805, 0.725] — 3 elemanlı bir dizi, tam grafikte X eksenindeki 3 noktaya karşılık gelen Y değerleri.

Basit ezber kuralı: axis=1 dediğinde, "her satırı kendi içinde özetle" demiş oluyorsun — sütun sayısı azalır (yok olur), satır sayısı korunur. Bizim ihtiyacımız da tam buydu: 10 max_depth değeri (satır) kalsın, her birinin 10 fold'u (sütun) tek ortalamaya insin.
'''
mean_train_score = np.mean(train_score, axis=1) #Her satır için, o satırın tüm sütunlarını bir grup say ve o grubun ortalamasını al.
mean_test_score = np.mean(test_score, axis=1) #test skorlarında belirgin bir düşüş, model ezberliyo çünkü #Her satır için, o satırın tüm sütunlarını bir grup say ve o grubun ortalamasını al.




cart_best_grid.best_params_['max_depth']
plt.plot(range(1, 11), mean_train_score,
         label="Training Score", color='b')

plt.plot(range(1, 11), mean_test_score,
         label="Validation Score", color='g')

plt.title("Validation Curve for CART")
plt.xlabel("Number of max_depth")
plt.ylabel("AUC")
plt.tight_layout()
plt.legend(loc='best')
plt.show()

#Fonksiyonlaştıralmı
def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)

val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")

cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]] # 0 0 -> max depth , # 0 1 -> range (1,11)  i=1,  1 0 -> min_samples_split , 1 1 -> range(2,20) , gözle takip et anlarsın forda

for i in range(len(cart_val_params)):
    val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])
    '''
    ┌──────────────────────────────────────────────┬───────────────────┐
│                    Katman                    │ Kaç kez çalışıyor │
├──────────────────────────────────────────────┼───────────────────┤
│ Dıştaki for döngüsü                          │ 2 kez (i=0, i=1)  │
├──────────────────────────────────────────────┼───────────────────┤
│ i=0 turunda validation_curve'ün iç eğitimi   │ 10 × 10 = 100     │
├──────────────────────────────────────────────┼───────────────────┤
│ i=1 turunda validation_curve'ün iç eğitimi   │ 18 × 10 = 180     │
├──────────────────────────────────────────────┼───────────────────┤
│ Toplam model eğitimi (tüm kod bloğu boyunca) │ 100 + 180 = 280   │
└──────────────────────────────────────────────┴───────────────────┘
    '''





# %%
################################################
# 8. Visualizing the Decision Tree
################################################
# [ref: cart.py:240]

# graphviz sistem binary'si (dot.exe) kurulu olmadığı için tree_graph() calismiyor.
# Alternatif: sklearn.tree.plot_tree (ekstra kurulum gerektirmez)

from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(cart_final, feature_names=list(X.columns), filled=True, ax=ax)
plt.show()
cart_final.get_params()


#Entropi için deneme
cart_entropy = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=17).fit(X, y)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(cart_entropy, feature_names=list(X.columns), filled=True, ax=ax)
plt.show()

# %%
################################################
# 9. Extracting Decision Rules
################################################
# [ref: cart.py:258]

tree_rules = export_text(cart_final,feature_names=list(X.columns))
print(tree_rules)











# %%
################################################
# 10. Extracting Python Codes of Decision Rules
################################################
# [ref: cart.py:266]

# skompiler Python 3.14'te calismiyor (ast.Num kaldirildi) -> m2cgen alternatifi
# pip install m2cgen

import m2cgen as m2c

print(m2c.export_to_python(cart_final))


# %%
################################################
# 11. Prediction using Python Codes
################################################
# [ref: cart.py:280]

def score(input):
    if input[1] <= 127.5:
        if input[7] <= 28.5:
            if input[5] <= 45.39999961853027:
                if input[5] <= 30.949999809265137:
                    var0 = [0.9867549668874173, 0.013245033112582781]
                else:
                    var0 = [0.8448275862068966, 0.15517241379310345]
            else:
                if input[2] <= 99.0:
                    var0 = [0.0, 1.0]
                else:
                    var0 = [1.0, 0.0]
        else:
            if input[5] <= 26.350000381469727:
                if input[5] <= 9.649999618530273:
                    var0 = [0.0, 1.0]
                else:
                    var0 = [1.0, 0.0]
            else:
                if input[1] <= 99.5:
                    var0 = [0.8181818181818182, 0.18181818181818182]
                else:
                    var0 = [0.5, 0.5]
    else:
        if input[5] <= 29.949999809265137:
            if input[1] <= 145.5:
                if input[4] <= 132.5:
                    var0 = [0.7857142857142857, 0.21428571428571427]
                else:
                    var0 = [1.0, 0.0]
            else:
                if input[7] <= 25.5:
                    var0 = [1.0, 0.0]
                else:
                    var0 = [0.41935483870967744, 0.5806451612903226]
        else:
            if input[1] <= 157.5:
                if input[7] <= 30.5:
                    var0 = [0.54, 0.46]
                else:
                    var0 = [0.27692307692307694, 0.7230769230769231]
            else:
                if input[4] <= 629.5:
                    var0 = [0.11235955056179775, 0.8876404494382022]
                else:
                    var0 = [0.6666666666666666, 0.3333333333333333]
    return var0

X.columns
input = [12, 1, 20, 23, 2, 2, 12, 7]
input = [6, 148, 70, 35, 0, 30, 0.62, 50]
score(input)





# %%
################################################
# 12. Saving and Loading Model
################################################
# [ref: cart.py:351]

#Modeli kurduk final model elimizde, her seferinde cart.py açıp çalıştırıp modeli kullanacak halimiz yok
#Bu noktada modeli saklamamız gerekiyor


joblib.dump(cart_final,'cart_final.pkl')
cart_model_from_disc = joblib.load('cart_final.pkl')
x = [12, 13, 20, 23, 4, 2, 12, 7]
cart_model_from_disc.predict(pd.DataFrame(x).T)