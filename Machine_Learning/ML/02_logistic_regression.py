# %%
######################################################################
# LOGISTIC REGRESSION
######################################################################
# Referans : machine_learning_miuul/logistic_regression.py
# Veri     : datasets/diabetes.csv
# Calistirma: proje kokunden, hucre hucre (Ctrl+Enter)
#
# !! BU DOSYADA DIKKAT:
#    - plot_roc_curve KALDIRILDI (sklearn 1.2) -> RocCurveDisplay.from_estimator kullan  [ref:45, 276]
######################################################################


# %%
# --- IMPORT'LAR ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.constants import pt
from sklearn.metrics._plot import roc_curve

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.metrics import RocCurveDisplay

from machine_learning_miuul.advanced_trees import cv_results
from machine_learning_miuul.logistic_regression import X_train, y_train

# %%
# --- ÖN TANIMLI FONKSİYONLAR ---

#%%
'''
'''
def outlier_thresholds(dataframe,col_name,q1=0.05,q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3-quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 *interquantile_range
    return low_limit,up_limit


'''
hangi dataframe'de , hangi kolonda aykırı değer var mı yok mu _? 
'''
def check_outlier(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False


'''
Aykırı değerleri düzeltme işlemi yaptığımız fonksiyon, ilgili sınırlara çeker aykırı değerleri
'''
def replace_with_thresholds(dataframe,variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable]<low_limit),variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
    #.loc[filtre,kolon] = değer --> loc sonucu True/False şeklinde döner dataset, ilgili kolonun filtre sonucu True dönen satırlarındaki hücrelerin değerleri güncellenir







pd.set_option('display.max_columns',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)
pd.set_option('display.width',500)









# %%
#Exploratory Data Analysis
# [ref: logistic_regression.py:2]


df = pd.read_csv('datasets/diabetes.csv')

df['Outcome'].value_counts() #Target(Bağımlı değişken) Analizi

sns.countplot(x='Outcome',data=df)
plt.show()


100 * df["Outcome"].value_counts() / len(df)
'''
(df['Outcome'].value_counts() / 768)*100   #768=len(df)
'''

##Feature Analizi (Bağımsız Değişkenler)

df.head()
df['BloodPressure'].hist(bins=20)
plt.xlabel('BloodPressure')
plt.ylabel('Number Of People')
plt.show()






def plot_numerical_col(dataframe,numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)


for col in df.columns:
    plot_numerical_col(df,col)

cols = [col for col in df.columns if 'Outcome' not in col]


for col in cols:
    plot_numerical_col(df,col)




df.describe().T









# %%
######################################################
# Target vs Features
######################################################

df.groupby('Outcome')['Insulin'].mean()
df.groupby('Outcome')['Pregnancies'].mean()

df.groupby('Outcome').agg({'Pregnancies' : 'mean'})

def target_summary_with_num(dataframe,target,numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end='\n\n\n')

for col in cols:
    target_summary_with_num(df,'Outcome',col)





# %%
##############################
# Data Preprocessing
##############################


df.shape
df.head()

df.isnull().sum()




#İlk iş olarak outlier değer olan feature'ları tespit ettik
for col in cols:
    print( col,check_outlier(df,col))
'''
Pregnancies False
Glucose True
BloodPressure True
SkinThickness False
Insulin True
BMI True
DiabetesPedigreeFunction False
Age False
'''

#Şimdi ise outlier'ları low ve up limitlerle değiştireceğiz daha iyi bir model için


for col in cols:
    if check_outlier(df,col):
        replace_with_thresholds(df, col)

#RobustScaler()
for col in cols:
    df[col]  = RobustScaler().fit_transform(df[[col]])
'''Logistic regression gibi modeller, büyük sayılır değişkenlere daha önemliymiş gibi
 ağırlık verme eğiliminde olabilir,bunu engellemek adına bütün veri setini ölçekleriz
'''

df.head()








# %%
##############################
#Veriyi Ölçeklendirdik şimdi Modeli kuralım
##############################


y= df['Outcome'] #Bağımlı değişkenimiz
X = df.drop(['Outcome'],axis=1) #Feature ' larımız , Bağımsız değişkenlerimiz

log_model = LogisticRegression().fit(X,y)

log_model.intercept_  # -- > BIAS  = -1.23367499
log_model.coef_  # --> Weights for each columns w1 w2 ..  [[ 0.60026858,  1.41864488, -0.2317767 ,  0.02249032, -0.14660392, 0.81662929,  0.35519151,  0.25514769]]

y_pred  = log_model.predict(X) #X bakğımsız değişkenlerini kullanarak tahminde bulun

y_pred[0:5]
y[0:5]  # ilk 5 çıktı doğru tahmin edilmiş








# %%
##############################
# Model Evaluation , iyi mi kötü mü değerlendirelim hata  ve başarı analizleriyle
##############################


#Modeli kurduk , tahmin ürettik, şimdi bu tahminler ne kadar iyi sorusuna cevap arıyoruz.
def plot_confusion_matrix(y,y_pred):
    acc = round(accuracy_score(y,y_pred),2)
    cm = confusion_matrix(y,y_pred) #Satırlar gerçek sınıf , Sütunlar tahmin edilen sınıf
    sns.heatmap(cm,annot=True,fmt='.0f') #annot-> her hücre içine sayıyı da yazdırıyor., fmt yazılan sayının formatı
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc),size=10)
    plt.show()


plot_confusion_matrix(y,y_pred)

print(classification_report(y, y_pred))
'''
plot_confusion_matrix(y,y_pred)
print(classification_report(y, y_pred))
              precision    recall  f1-score   support
           0       0.80      0.89      0.84       500
           1       0.74      0.58      0.65       268
    accuracy                           0.78       768
   macro avg       0.77      0.74      0.75       768
weighted avg       0.78      0.78      0.78       768

'''



#ROC AUC
#NEDİR?
'''
Şimdiye kadar predict() ile modelin kesin kararını (0 ya da 1) kullandık.
Ama logistic regression aslında arka planda önce bir olasılık hesaplıyor
(mesela "%73 ihtimalle diyabet"), sonra bunu 0.5 eşiğine göre 0 ya da 1'e yuvarlıyor.
ROC-AUC, işte bu ham olasılıkları kullanarak modelin sınıfları ne kadar iyi ayırt edebildiğini
ölçüyor — eşik değerinden (0.5) bağımsız olarak.
'''
#NEDEN ONEMLI?
'''
Accuracy/precision/recall tek bir eşik (0.5) için hesaplanır.
Ama belki 0.5 yerine 0.3 eşiği kullansan model daha iyi ayrım yapardı
(özellikle bizim durumumuzdaki gibi, recall'u düşük bir modelde).
ROC-AUC, tüm olası eşikleri tarayıp modelin genel ayırt etme gücünü tek bir sayıda özetler:
1.0 = mükemmel ayrım, 0.5 = rastgele tahminden farksız.
'''

y_prob = log_model.predict_proba(X)[:,1]
round(roc_auc_score(y,y_prob),2)  #0.84
#Uzun açıklama
'''
y = df['Outcome']           # hedef değişken (bağımlı)
X = df.drop(['Outcome'],axis=1)   # geri kalan 8 kolon (bağımsız değişkenler / feature'lar)

Yani X, Outcome hariç tüm kolonları (Pregnancies, Glucose, BloodPressure...) içeren bir tablo — modelin tahmin yaparken girdi olarak kullandığı değerler. predict_proba(X) dediğinde, bu 8 feature'ı kullanarak her satır için olasılık üretiyor.

<br>

[:,1] kısmı — somut örnekle gidelim:

predict_proba(X) çıktısı, her satırda 2 sayı olan bir tablo (numpy array) döndürüyor. Diyelim ilk 3 hasta için çıktı şöyle:

         [0 olasılığı, 1 olasılığı]
Hasta 1: [0.70,        0.30]
Hasta 2: [0.15,        0.85]
Hasta 3: [0.60,        0.40]

Bu, 2 boyutlu bir yapı (satır x sütun). [:,1] diyerek şunu söylüyorsun: "tüm satırları al (:), ama sadece index 1'deki sütunu al (1)" — yani sadece sağdaki (1 olasılığı) sütunu:

Hasta 1: 0.30
Hasta 2: 0.85
Hasta 3: 0.40


'''







# %%
######################################################
#Model Validation ama HOLDOUT YÖNTEMİ İLE
######################################################
#  Feature  | Target
#  X_train    y_train     #EĞİTİM
#  X_test     y_test      #TEST



'''
 train_test_split fonksiyonun çıktı sırası vardır ve sabittir,
 bizim de çıktıları bu sırada yakalamamız gerekir

'''
X_train, X_test, y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=17)


log_model = LogisticRegression().fit(X_train,y_train) #Train setine göre eğittik
y_pred = log_model.predict(X_test) #test setine göre tahmin yaptık
y_prob = log_model.predict_proba(X_test)[:,1] # ROC AUC kontrolü test seti için

print(classification_report(y_test,y_pred))

RocCurveDisplay.from_estimator(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()


roc_auc_score(y_test, y_prob)












# %%
######################################################
# Model Validation: 10-Fold Cross Validation
######################################################


y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y)

'''Holdout yönteminde (bir önceki adım) veriyi tek bir kez train/test diye ikiye böldük
ama bu bölünme rastgele, random_state'e göre değişebilir, yani şansa bağlı bir sonuç riski var
(test setine hangi hastaların düştüğü sonucu etkileyebilir, tıpkı biraz önceki:
AUC 0.84 vs 0.88 farkında gördüğümüz gibi).
Cross validation, bu şans faktörünü azaltmak için veriyi birden fazla kez, farklı şekillerde böler,
her seferinde ayrı bir test seti kullanır, sonra hepsinin ortalamasını alır
 daha güvenilir bir sonuç

''' #cv stands for cross_validate btw
cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327


