# %%
######################################################################
# KNN
######################################################################
# Referans : machine_learning_miuul/knn.py
# Veri     : datasets/diabetes.csv
# Calistirma: proje kokunden, hucre hucre (Ctrl+Enter)
######################################################################
import random
from idlelib.autocomplete import AutoComplete

# %%
# --- IMPORT'LAR ---

import pandas as pd
from sklearn.metrics import classification_report,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from machine_learning_miuul.knn import X_scaled, knn_model, cv_results, knn_gs_best

pd.set_option('display.max_columns',None)
pd.set_option('display.width',500)






# %%
#Data import
#EDA

df= pd.read_csv('datasets/diabetes.csv')

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()





# %%
################################################
# Data PreProcessing / Feature Engineering
################################################
# [ref: knn.py:2]


y=df['Outcome']
X=df.drop(['Outcome'],axis=1)

X_scaled =StandardScaler().fit_transform(X) #Feature'ları standartlaştırdık

X = pd.DataFrame(X_scaled,columns=X.columns) #Featurelar artık standartlaştırıldı


# %%
################################################
# 3. Modeling & Prediction
################################################
# [ref: knn.py:45]
#ÖNEMLİ BİR FARKINDALIK NOTU
#LAZYLEARNER EAGER LEARNER OLAYI
'''
"eager learner" vs "lazy learner".

Diğer modeller (linear regression, logistic regression, CART...) → eager learner
fit() aşamasında asıl matematiksel iş bitiyor:
- Linear/logistic regression → fit() sırasında katsayılar (β değerleri) hesaplanıyor (gradient descent veya normal equation ile).
- CART → fit() sırasında ağacın dal/yaprakları, hangi değişkenden hangi eşikte bölüneceği hesaplanıyor.

predict() aşamasında ise sadece öğrenilmiş formülü/kuralı uyguluyorsun — hızlı, hafif bir işlem. Mesela linear regression'da predict = y = β0 + β1*x1 + ... formülüne yeni x'i koymak, o kadar.

KNN → lazy learner
fit() aşamasında gerçekten hiçbir hesap yapılmıyor, sadece veri ezberleniyor (X, y hafızaya alınıyor). Asıl iş predict()'e erteleniyor:
- Yeni gelen noktayla (random_user), eğitim setindeki her bir noktaya olan mesafe hesaplanıyor (Euclidean distance).
- Mesafeler küçükten büyüğe sıralanıyor.
- En yakın k (default 5) komşu seçiliyor.
- Bu komşuların çoğunluk sınıfına göre tahmin (0/1) veriliyor.
'''
knn_model = KNeighborsClassifier().fit(X,y)

random_user  = X.sample(1,random_state=45)

knn_model.predict(random_user)




# %%
################################################
# Confusion Matrix İçin y pred + AUC İÇİN y_prob ardından da değerlendirmeler
################################################
# [ref: knn.py:55]

y_pred = knn_model.predict(X)

y_prob = knn_model.predict_proba(X)[:,1]

print(classification_report(y,y_pred))
# f1 0.74
#acc 0.83

#AUC
roc_auc_score(y,y_prob) #0.9017

#CROSS VALIDATION YAPALIM
#BIR DE BOYLE KONTROL EDELIM



#NEDEN VE NASIL CROSS VALIDATION YAPIYORUZ?
'''
Buraya kadarki üç satır bir sorun taşıyor: model X, y ile eğitildi, sonra yine aynı X, y üzerinde test edildi.
Bu "in-sample" değerlendirme — model gördüğü veriyi tahmin ediyor, bu yüzden skorlar gerçekte olduğundan iyimser (fazla yüksek) çıkabilir. 
Tam da bu yüzden bir sonraki adımda cross-validation'a geçiliyor:

cv_results = cross_validate(knn_model,X,y,cv=5,scoring=['accuracy','f1','roc_auc'])
- cv=5 → veriyi 5 parçaya (fold) bölüyor, her seferinde 4 parçayla eğitip 1 parçayla test ediyor, bunu 5 kez tekrarlıyor (her fold sırayla test seti oluyor).
- scoring=[...] → her fold için aynı anda 3 farklı metriği (accuracy, f1, roc_auc) hesaplattırıyoruz.
- knn_model burada zaten fit edilmiş olsa da önemli değil — cross_validate içeride kendi fold'larıyla modeli sıfırdan yeniden eğitip test ediyor,
sana sadece modelin "tarifini" (hangi sınıf, hangi parametreler) veriyor.
- Sonuç cv_results, bir sözlük (dict) — her metrik için 5 fold'un skorlarını dizi halinde tutuyor.




'''

cv_results  = cross_validate(knn_model,X,y,cv=5,scoring=['accuracy','f1','roc_auc'])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()
#Aklıma takılan 3 soruya ilişkin cevap
'''
1. "Cross-validation" vs "cross_validate" — mentörün ayrımı

- Cross-validation (çapraz doğrulama) → bir kavram/yöntem. "Veriyi k parçaya böl, sırayla her parçayı test seti yap, kalanla eğit" fikri. Bu, hiçbir kod değil, genel bir değerlendirme stratejisi.
- cross_validate → sklearn'ün bu kavramı uygulayan fonksiyonlarından biri. sklearn'de bu kavramı hayata geçiren birkaç farklı fonksiyon var:
  - cross_val_score → sadece tek bir metrik, düz bir array döner (en basit hali).
  - cross_val_predict → her fold'da yapılan tahminleri birleştirip döner (skor değil, tahmin).
  - cross_validate → birden fazla metriği aynı anda hesaplar, üstüne fit/score sürelerini de ekler, sözlük (dict) döner (en detaylı hali — bizim kullandığımız bu).

Yani "cross-validation" şemsiye kavram, "cross_validate" ise o kavramı uygulayan spesifik sklearn fonksiyonu — mentörün ayırdığı tam bu.

2. scoring=[...] içine biz mi yazıyoruz?

Evet, tamamen sen belirliyorsun. scoring parametresine, sklearn'ün önceden tanımlı metrik isimlerinden (string olarak) istediğin kadarını liste halinde veriyorsun: 'accuracy', 'f1', 'roc_auc', 'precision', 'recall' vs. hepsi seçenek. Sen 3 tanesini istedin, o yüzden çıktıda 3 tanesi var — 5 tane isteseydik çıktıda 5 tane olurdu.

3. Çıktının formatı (cv_results)

cv_results bir Python dict'i. Her key bir string, her value ise bir NumPy array — array'in uzunluğu cv=5 dediğin için 5 (her fold'un skoru ayrı ayrı):

{
  'fit_time':        array([...5 sayı...]),   # her fold'da modeli eğitmek kaç saniye sürdü
  'score_time':       array([...5 sayı...]),   # her fold'da test/skorlama kaç saniye sürdü
  'test_accuracy':    array([...5 sayı...]),   # senin istediğin metrik #1, her fold için
  'test_f1':          array([...5 sayı...]),   # senin istediğin metrik #2, her fold için
  'test_roc_auc':     array([...5 sayı...]),   # senin istediğin metrik #3, her fold için
}

- fit_time ve score_time sen istemesen de otomatik gelir — sadece performans/hız bilgisi, göz ardı edebilirsin.
- test_accuracy, test_f1, test_roc_auc → isimlerin başına test_ eklenmesi sklearn'ün varsayılan davranışı: sadece test (her fold'da eğitilmeyen, tutulan) parçadaki skoru veriyor. Eğer cross_validate(..., return_train_score=True) deseydin, ayrıca train_accuracy, train_f1 gibi anahtarlar da gelirdi (eğitim setindeki skor — overfitting kontrolü için kullanılır, biz istemedik, o yüzden yok).
- Her array'in 5 elemanı olması = cv=5 dediğin için 5 fold var, her fold ayrı bir sayı üretti. Sonra sen .mean() ile bu 5 sayının ortalamasını alıp tek bir özet sayıya indiriyorsun.
'''



knn_model.get_params()



# %%
################################################
# 5. Hyperparameter Optimization
################################################
# [ref: knn.py:90]


#Kullanıcıların dışarıdan müdahale ederek ayarlaması gereken Hiperparametreler vardır.
#En iyi komşuluk hiperparamteresini algoritma ile deneyerek buluruz

knn_model = KNeighborsClassifier()
knn_model.get_params() #{'algorithm': 'auto', 'leaf_size': 30, 'metric': 'minkowski', 'metric_params': None, 'n_jobs': None, 'n_neighbors': 5, 'p': 2, 'weights': 'uniform'}
#'n_neighbors': 5   peki 5 en iyisi mi? Denemeden bilemeyiz

knn_params = {'n_neighbors':range(2,50)}
knn_gs_best = GridSearchCV(knn_model, knn_params, cv=5, n_jobs = -1,verbose= 1).fit(X,y)
knn_gs_best.best_params_ #17


#Büyük resimde kafa karışıklığı yaşanırsa kaybolmamak adına net bir tablo
'''
1. Preprocessing → veriyi standartlaştır (StandardScaler)
2. KNN ile direkt model eğit → default ayarla (n_neighbors=5)
3. cross_validate ile ölç → 5 fold'da test et, ortalama skor al (acc/f1/auc) → "default ayar bu kadar başarılı" cevabını al
4. Soru sor: "farklı komşu sayısıyla daha iyi olur mu?" → GridSearchCV devreye girer, verilen range'deki (2-49) her komşu değerini dener, her biri için içeride cross-validation yapar, en iyi ortalamayı veren k'yı (best_params_) döndürür
5. Bulunan en iyi hiperparametreyi (k=17) tekrar cross_validate'e ver → son bir doğrulama, "gerçekten k=17 ile skorlar iyileşmiş mi" diye kontrol
6. Son adım: final modeli bu en iyi hiperparametreyle eğit → artık kullanıma hazır, gerçek model bu



'''




# %%
################################################
# 6. Final Model
################################################
# [ref: knn.py:107]

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X,y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv=5,
                            scoring=['accuracy','f1','roc_auc'])


cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)
knn_final.predict(random_user)













