import numpy as np
import pandas as pd
from numpy.ma.core import negative

numbers = [1,2,2,3,4,4,5]
unique_numbers={x for x in numbers}
print(unique_numbers)


numbers = [1,2,3,4,5]
new_numbers=[x for x in numbers if x%2==0]
print(new_numbers)

np.zeros(5)
np.array([1,2,3,4,5])
np.empty((3,3,3,3))

np.array([1,2,3,4,5]).mean()



genc<30
30<=ortayas<=60
yaslı>60

import pandas as pd
veri = {
              'İsim': ['Ahmet', 'Mehmet', 'Ayşe', 'Fatma', 'Ali'],
               'Yaş': [25, 35, 60, 55, 70]
}
df = pd.DataFrame(veri)

df['Yaş Kategorisi'] = df['Yaş'].apply(lambda x: 'Genç' if x < 30 else ('Orta yaşlı' if x <= 60 else 'Yaşlı'))



# İpucu: lambda ifadesi ve apply yöntemini kullanın.




import matplotlib.pyplot as plt

notlar = [68, 74, 82, 90, 78, 85, 92, 88, 76, 61, 79, 73, 89, 81, 72, 95, 70, 83, 77, 75]

plt.hist(notlar, bins=10, edgecolor='r', alpha=0.7)
plt.xlabel('Notlar')
plt.ylabel('Frekans')
plt.title('Sınav Notları Dağılımı')
plt.show()


Ekrana gösterilecek histogram grafiğinde kaç adet çubuk olacaktır ?





def test(x,y):
    print(x*y)


sonuc=test(3,5)
print(sonuc)
print(test(3,5))

t={1,2,3,4}
{x:x**2 for x in t if x<5}



v = np.array([5, -3, 10, -8, 0, -1, 4])
negatives = v[v<0]
positives = v[~v<0]

print(negatives)
print(positives)
print(f"{negatives},{positives}")

