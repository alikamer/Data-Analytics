# %%
import matplotlib
import os
import numpy as np
import pandas as pd
import seaborn as sns
from fontTools.misc.symfont import printGreenPen
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date

from pandas.core.array_algos import quantile
from pandas.core.interchange import dataframe
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler

# from Downloaded.feature_engineering import age_index, new_df, na_cols

# %%
# --- BURADAN AŞAĞISI KOPYA DEMİRBAŞLAR (TEK SEFERDE ÇALIŞTIRMAK İÇİN) ---

def load_application_train():
    data = pd.read_csv("dataset/application_train.csv")
    return data

def load():
    data = pd.read_csv("dataset/titanic.csv")
    return data

def outlier_threshold(dataframe,col_name,q1=0.25,q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    iqr = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * iqr
    low_limit = quantile1 - 1.5 * iqr
    return low_limit,up_limit

def check_outlier(dataframe,col_name):
    low_limit,up_limit=outlier_threshold(dataframe,col_name)
    if dataframe[(dataframe[col_name]<low_limit)|(dataframe[col_name]>up_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype.name in ["object", "category", "string", "str"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtype.name not in ["object", "category", "string", "str"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtype.name in ["object", "category", "string", "str"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype.name not in ["object", "category", "string", "str"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def grab_outliers(dataframe,col_name,index=False):
    low, up = outlier_threshold(dataframe,col_name)
    if dataframe[((dataframe[col_name]<low)) | ((dataframe[col_name]>up))].shape[0]>10:
        print(dataframe[((dataframe[col_name]<low)) | ((dataframe[col_name]>up))].head())
    else :
        print(dataframe[((dataframe[col_name] < low)) | ((dataframe[col_name] > up))])
    if index:
        outlier_index=dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].index
        return outlier_index

def remove_outliers(dataframe,col_name):
    low,up = outlier_threshold(dataframe,col_name)
    df_without_outliers=dataframe[~((dataframe[col_name]<low) | (dataframe[col_name]>up))]
    return df_without_outliers

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

# -----------------------------------------------------------------------------

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x:'%.3f' % x)
pd.set_option('display.width',500)

# %%
def load_application_train():
    data = pd.read_csv("dataset/application_train.csv")
    return data

df = load_application_train() # Bu yükleme 38. satırda titanic yüklenirken ezildiği için performansı artırmak adına kapatıldı.
df.head()

# %%
def load():
    data = pd.read_csv("dataset/titanic.csv")
    return data


df = load()
df.head()
# %%
##Grafik tekniği ile ayıkırı değerler


sns.boxplot(x=df['Age'])
plt.show()


q1=df['Age'].quantile(0.25)
q3=df['Age'].quantile(0.75)

iqr=q3-q1
up=q3 + 1.5 * iqr
low=q1-1.5 * iqr


df[(df['Age']<low) | (df['Age']>up)] ##aykırı değerleri getitirirz, age den büyük ve küçük olanlardır iqr hesabı sonucu bwelirlenen low ve up limitlere göre


df[(df['Age']<low) | (df['Age']>up)].index #ile ilgili değerlerin indexlerine ulaşırız



##Sadece T/F döndürerek outlier kontrolü  .any

print(df[(df['Age']<low) | (df['Age']>up)].any(axis=None))

(df['Age']<low).any(axis=None) ## eksi yaş yoktur, false döner

##low up eşik değer belirlendi
#Outliers bulundu


##Sıra işlemleri fonksiyonlaştırıp,bütün sütunlarda gezmede

#ilk olarak outlier_threshold adında low ve up sınırları oto belirleyeceğimiz bir func yazalım
# %%
def outlier_threshold(dataframe,col_name,q1=0.25,q3=0.75):
    quantile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    iqr = quantile3 - quantile1
    up_limit = quantile3 + 1.5 * iqr
    low_limit = quantile1 - 1.5 * iqr
    return low_limit,up_limit


low,up=outlier_threshold(df,'Age') #uplowları oto alıyoruz artık şimdi diğer adım

#outlier_threshold fonoksiyonunu da kullanarak yeni bir fonksiyon yazalım ve direkt ilgili değişkenin outlier içerip içermediğini anlayalım
# %%
def check_outlier(dataframe,col_name):
    low_limit,up_limit=outlier_threshold(dataframe,col_name)
    if dataframe[(dataframe[col_name]<low_limit)|(dataframe[col_name]>up_limit)].any(axis=None):
        return True
    else:
        return False
# %%
#nunuqiue --> number of unique döner  , unique---> unique döner
check_outlier(df,'Age')
check_outlier(df,'Fare')
##şimdi yüzlerce değişken varken napcaz? onun içinde fonksiyon yazalım, tek tek elle yazamayız

dff = load_application_train()
dff.head()
# %%
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype.name in ["object", "category", "string", "str"]]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtype.name not in ["object", "category", "string", "str"]]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtype.name in ["object", "category", "string", "str"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype.name not in ["object", "category", "string", "str"]]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
# %%
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)


#---------------------------------------------------------------------------------------------------------------------
#Aykırı değere erişmek için fonksiyon yazalım
#%%
def grab_outliers(dataframe,col_name,index=False):
    low, up = outlier_threshold(dataframe,col_name)

    if dataframe[((dataframe[col_name]<low)) | ((dataframe[col_name]>up))].shape[0]>10:
        print(dataframe[((dataframe[col_name]<low)) | ((dataframe[col_name]>up))].head())
    else :
        print(dataframe[((dataframe[col_name] < low)) | ((dataframe[col_name] > up))])

    if index:
        outlier_index=dataframe[((dataframe[col_name]<low)|(dataframe[col_name]>up))].index
        return outlier_index
#%%
grab_outliers(df,'Age',True)
age_index=grab_outliers(df,'Age',True)
age_index

##Kısaca 3 şey yaptık
outlier_threshold(df,'Age')
check_outlier(df,'Age')
grab_outliers(df,'Age',True)


#-----------------------------------------------------------------------------------------

#Aykırı Değer Problemini Çözme

#1-Silme

low,up=outlier_threshold(df,'Fare') # Fare (Bilet Ücreti) değişkeni için alt ve üst limitleri hesaplar
df.shape # Veri setinin orijinal boyutunu (satır, sütun) verir
df[(df['Fare']<low)|(df['Fare']>up)].shape # Aykırı (outlier) olan satırların boyutunu verir.

df[~((df['Fare']<low)|(df['Fare']>up))].shape # Aykırı olmayan (normal) satırların boyutunu verir


##Üstte tek bir değişkenm için aykırılıkları seçtik, ama biz bütün değşikenler için aykırılıkları seçip sornasında silmek istersek func yazmamız gerekiyor

#%%
##Önemli Not --> Outliers numericler üzerinde aranır
def remove_outliers(dataframe,col_name):
    low,up = outlier_threshold(dataframe,col_name)
    df_without_outliers=dataframe[~((dataframe[col_name]<low) | (dataframe[col_name]>up))]
    return df_without_outliers
#%%
cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape
new_df.shape
for col in num_cols:
    new_df = remove_outliers(df, col)

df.shape[0] - new_df.shape[0]  #116 tane outlier gözlem birimi(satır) silindi
##Burada önemli bir nokta var, bir hücredeki outlier değerden dolayı
# aynı gözlem birimindeki diğer sağlam yani outlier olmayan değerlerde siliniyor.

#!!!!  İşte tam bundan dolayı bazı senaryolarda bu gözlem birimlerini silmek yerine, baskılamayıda tercih edebiliriz



## re-assignment with thresholds ( Aykırı değerleri baskılama yöntemi )
#Veri kaybetmemek adına, outlier değerler tespit edildikten sonra, th değerlerle değiştirilir

low,up = outlier_threshold(df,'Fare')
df[(df['Fare']<low) | (df['Fare']>up)]['Fare'].shape
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

df.loc[(df["Fare"] > up), "Fare"] = up

df.loc[(df["Fare"] <low>), "Fare"] = low


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_threshold(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape


for col in num_cols:
    print(col,check_outlier(df,col))


for col in num_cols:
    replace_with_thresholds(df,col)


for col in num_cols:
    print(check_outlier(df,col))

# Özetle outlier olanları low ve up ile değiştirkdik ilgililer ilgililere







##

#------------------------------------------------

#Çok değişkenli aykırıdeğer ANALİZİ


#local outlier factor  #LOF




# 17 OK 3 OK AMA 17 YAŞINDA OLUP 3 KERE EVLENMİŞ OLMAK OUTLİERDIR

#LOF yöntemide çok değişkenli outlier belirleme yöntemidir


df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64','int64'])
df = df.dropna()
df.head()
df.shape # 53940,7

for col in df.columns:
    print(col,check_outlier(df,col))

low,up = outlier_threshold(df,'carat')

df[((df['carat'] < low) | (df['carat'] > up))].shape #1889,7

low, up = outlier_threshold(df, "depth")
df[((df["depth"] < low) | (df["depth"] > up))].shape #2545,7

#LOF modelini tanımladık
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df) ##modelin mutlaka fit edilmesi gereklidir / modelin standart belirlediği kesin -1,1 değerleridir, ya outlier ya da inlier diye kesin şekilde kefeye ayırır

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5]


scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0,50], style='.-')
plt.show()

th = np.sort(df_scores)[3]  # skorları küçükten büyüğe sıraladık 4.skor olan -4.98415175 değerini th olarak belirledik

df[df_scores<th] #41918 48410 49189 bu 3 gözlem birimi sıkıntılı
df[df_scores<th].shape #3 gözlem birimi eşik değerin altında yani üstünde , outlier

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index #dademinki 3 indexi döndürdü

df[df_scores < th].drop(axis=0, labels=df[df_scores<th].index)


df = df[df_scores >= th]

df[df_scores<th]
df.index.isin([41918, 48410, 49189]).any()





#############################################

# Missing Values (Eksik Değerler)

#############################################
df=load()
df.head()

df.isnull().values.any()

df.isnull().sum() #boş değerlerin toplam sayısnı verir
df.notnull().sum()
df.notnull().sum()+df.isnull().sum()

#veri setindeki toplam boş değer sayısı
df.isnull().sum().sum()


df[df.isnull().any(axis=1)] #eksik rowları getirir, en az bir
df.loc[df.isnull().any(axis=1), df.isnull().any(axis=0)] #ilgili satırın sadece eksik olan hücresinin ait olduğu sütunları getirir, boşa dolularıda getirmez

df[df.notnull().all(axis=1)] #eksik olmayan rowları getirir, en az bir
df.loc[df.notnull().all(axis=1),df.notnull().all(axis=0)]

df.isnull().any(axis=1).sum().sum()

df.isnull().sum(axis=0)

df.isnull().sum().sort_values(ascending=False)

((df.isnull().sum()/df.shape[0])*100).sort_values(ascending=False)


na_cols=[col for col in df.columns  if df[col].isnull().sum()>0]



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)



















