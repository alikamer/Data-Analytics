# %%
######################################################################
# LINEAR REGRESSION
######################################################################
# Referans : machine_learning_miuul/linear_regression.py
# Veri     : datasets/advertising.csv
# Calistirma: proje kokunden, hucre hucre (Ctrl+Enter)
######################################################################


# %%
# --- IMPORT'LAR ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fontTools.ttLib import woff2

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score

from machine_learning_miuul.linear_regression import initial_b, learning_rate, initial_w, num_iters, cost_history

#ince ayar yapalım
pd.set_option('display.float_format',lambda x: '%.2f' %x)


# %%
# --- VERI YUKLEME ---

df = pd.read_csv('datasets/advertising.csv')
df.head()
#-----
#Since we gonna build a simple model with 2 parameters (for now)
X= df[['TV']]
y=df[['sales']] #

# %%
######################################################
# Modeli Kuralım ve inceleyelim b ve w değerlerini,
######################################################

reg_model =LinearRegression().fit(X,y)

#y_hat = b +w*TV

#sabit -b bias

reg_model.intercept_[0]

#tv'nin katsayısı w1

reg_model.coef_[0]
reg_model.coef_[0][0]

'''
Bizim tablomuz bu şekilde 
sadece tv ye bakarak sales tahmini yapıyoruz.
intercept için [0]
coefficent w için [0][0] eleman vardır 
┌──────────────┬──────┬────────┐
│              │  b   │  w_TV  │
├──────────────┼──────┼────────┤
│ sales modeli │ 7.03 │ 0.0475 │
└──────────────┴──────┴────────┘
intercept_ = [7.03] (1 eleman), coef_ = [[0.0475]] (1 satır, 1 sütun) 
— sen sadece o tek satırın tek sütununu okuyorsun: coef_[0][0].
'''


# %%
######################################################
# Simple Linear Regression with OLS Using Scikit-Learn
######################################################
# [ref: linear_regression.py:18]

#Tahmin

#150 birimlik tv harcaması olsa ne kadar satış olur?

#Temel formülü uygularız
# b+wxi
# burada xi değeri 150 birim olarak vreildi
# b ve değerlerini hesaplamıştık zaten  tek bir sütunumuz vardı

reg_model.intercept_[0] + reg_model.coef_[0][0]*150  #14 birimlik sale beklenir

#Peki 500 birimlik bir tv harcamasında kaç birim sale beklenir?
df.describe().T #görüyoruzki  verisetimizdeki maksimum tv harcaması  296, max sale ise 27 birim
reg_model.intercept_[0] + reg_model.coef_[0][0]*500  #30 birimlik sale beklenir


# %%
##############################
# Modelin Görselleştirilmesi
##############################

g=sns.regplot(x=X,y=y, scatter_kws={'color': 'b', 's': 9},
              ci=False, color='r')

g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0],2)} + TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel('SALES')
g.set_xlabel('TV Harcamaları')
plt.xlim(0,310)
plt.ylim(bottom=0)
plt.show()


# %%
##############################
# Şimdi Modelin yaptığı tahminlerin Başarısını değerlendirelim
##############################
# [ref: linear_regression.py:44]
y_pred=reg_model.predict(X)
#MSE
mean_squared_error(y,y_pred) # çıkan sonuç 10 fakat bu 10 ne anlama geliyor?? bir fikrimiz yok e napıcaz
#RMSE
np.sqrt(mean_squared_error(y,y_pred)) #3.2
#MAE
mean_absolute_error(y,y_pred) #2.5
#bu durumda mean ve std bakılır yorumlanır, 10 iyi mi kötü mü diye


#satış için yani bağımlı değişken için bakalım o zaman

y.mean() #14
y.std() #5

# mse değeri 10 idi, karekökünü alalım 3,2 küsür değerler 9 ile 19 arasında olması gerekirken 3 hata payı yüksektir.



#R-KARE çok önemli

reg_model.score(X,y)

#bu metrik şunu ifade eder; veri setindeki bağımsız değişkenlerin, bağımlı değişkenleri açıklama yüzdesidir.
#burada bağımsız değişken olan tv değişkenleri bağımlı değişken olan sale değişkenlerini yüzde 61 oranında açıklayabilmektedir


''' KAYBOLMAMAK ADINA


Genel özet — zincir bu şekilde:
1. X, y → ham veriyi ayır
2. .fit(X,y) → formülle (türev/normal equation) minimum hatayı veren sabit b, w'yi hesapla
3. .predict(X) → bu sabit b, w'yi her satırın kendi TV değeriyle çarpıp tahmin üret
4. mean_squared_error(y, y_pred) → üretilen tahminler gerçek değerlerden ne kadar sapmış, ölç
'''


# %%
######################################################
# Multiple Linear Regression
######################################################
# [ref: linear_regression.py:94]

df=pd.read_csv('datasets/advertising.csv')

X = df.drop('sales',axis=1)

y=df[['sales']]









# %%
##############################
# #Modeli kuralım
##############################
# [ref: linear_regression.py:105]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


reg_model = LinearRegression().fit(X_train,y_train)


reg_model.intercept_ #bias 2.9

reg_model.coef_ #w1 w2 w3   0.0468431 , 0.17854434, 0.00258619






# %%
##############################
# Tahmin
##############################
# [ref: linear_regression.py:123]

#Gözlem değerlerine göre satışın beklenen değeri nedir? basitçe hesapla
'''
TV:30
radio:10
newspaper:40
b-2.9
w1 w2 w3 ,  0.0468431 , 0.17854434, 0.00258619
'''

b=2.9
w1=0.0468431
w2=0.17854434
w3=0.00258619

b+30*w1+10*w2+40*w3  # Sales = 2.9 +TV*0.04 + radio*0.17 + newspaper*0.002

#Şimdide otomatik hesaplayalım predict ile

tahmin_1 = [[30],[10],[40]]
tahmin_1 = pd.DataFrame(tahmin_1).T

reg_model.predict(tahmin_1) #tahmin sonucu 6.2










# %%
##############################
# Tahmin Başarısını Değerlendirme
##############################
# [ref: linear_regression.py:145]

#TRAIN RMSE
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train,y_pred))

#TRAIN RKARE
reg_model.score(X_train,y_train)


#test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test,y_pred))
#1.41

#Test RKARE
reg_model.score(X_test,y_test)


#10 KATLI CV RMSE

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring='neg_mean_squared_error')))
#1.69 - 10









# %%
######################################################
# Simple Linear Regression with Gradient Descent from Scratch
######################################################
# [ref: linear_regression.py:187]

def cost_function(Y,b,w,X):
    m = len(Y) #değişken gözlem birimi adedi, ortalama için
    sse = 0 #summation

    for i in range(0,m):
        y_hat = b+w* X[i]   # y_hat yani tahminler b+w * bağımsız deişkenler sırayla  hepsi itere edilecek
        y = Y[i] # gerçek tahmin değerleri

        sse += (y_hat-y)**2   #summation güncellenir her seferinde, hatalar farkının kareleri toplanarak, formülü uyguluyoruz adım adım basit yani


    mse = sse/m   #mean square error hesaplanır /m  ile
    return mse

def update_weights(Y,b,w,X,learning_rate):
    m = len(Y)
    b_deriv_sum = 0  #ilgili formülde b v w yani theta1,theta0 değerlerinin güncellenmesi için kısmi türev kısmına geldik,
    w_deriv_sum = 0  #'''

    for i in range(0,m):
        y_hat = b + w*X[i] #aynı işlem, her gözlem birimi için i artar yani satr ilerler, ilgili bağımsız değişken için bağımlı değişken tahmini yapılır ve y_hat olarak güncellenir
        y = Y[i]           # aynı zamanda, paralel olarak ilgili gözlem birimine (i) gidlilri ve o anki gerçek y değeri de alınır
        b_deriv_sum += (y_hat-y)
        w_deriv_sum += (y_hat-y) * X[i]        #biz zaten b ve w değerleri için kısmi türevlerin alınmış halini yazdık, türev alma işlemini kafadan yaptık kısaca burada öyle bir işlem yapmıyoruz gerekte oyk

    # son kısım olan b,w değerlerinin güncellenmesine geliyoruz   b=b-eğim mantığı ama learning rate var,


    new_b = b-(learning_rate * 1/m * b_deriv_sum)   #hatalar toplandı ortalaması alındı learning rate ile çarpıldı kısaca kısmi türev uygulandı, kabaca değişim miktarı bulundu
    new_w  = w-(learning_rate * 1/m * w_deriv_sum)
    return new_b, new_w

def train(Y,initial_b,initial_w,X,learning_rate,num_iters):
   # print('Starting gradient descent at b={0}, w{1}, mse = {2}'.format(initial_b,initial_w,cost_function(Y,initial_b,initial_w,X)))   #Hoca .format ile yazmış
    print(f'Starting gradient descent at b={initial_b}, w={initial_w}, mse={cost_function(Y,initial_b,initial_w,X)}')       #fstring daha pratik

    b = initial_b  #kısaltmalar  yaptık daha pratik olsun diye
    w = initial_w
    cost_history =[]  # burada tutacağız kayıtları, her bir cost sonucunu görmek adına ilgili w,b değerleri için

    for i in range(num_iters):
        b,w = update_weights(Y,b,w,X,learning_rate)
        mse = cost_function(Y,b,w,X)
        cost_history.append(mse)

        if  i%100 == 0:  # her  100 katında sonuç göstersin
            print(f'iter={i:d}  b={b:.3f}  w={w:.3f}  mse={mse:.4}')

    print(f'After {num_iters} iterations b={b}  w={w} mse={cost_function(Y, b, w, X)}')
    return  cost_history,b,w


df= pd.read_csv('datasets/advertising.csv')
X= df['radio']
Y= df['sales']

#hiperparametreleri ayarlayalım (kullanıcının belirlediği demek, veri setinde olmayan)
learning_rate = 0.001
initial_b = 0.001
initial_w = 0.001
num_iters = 10000


cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)




from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(df[['radio']], df[['sales']])
y_pred = reg.predict(df[['radio']])
mean_squared_error(df['sales'], y_pred)


##BİRDEN ÇOK BAĞIMSIZ DEĞİŞKEN İÇİN MODELİ TEKRAR KUR BAKALIM SONUÇ DÜŞECEK Mİ!!! BU NOTU Aİ GÖRÜRSE HATIRLATSIN
