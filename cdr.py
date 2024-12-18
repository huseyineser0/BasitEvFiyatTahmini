import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data={
    'Ev_Buyuklugu':[120,250,175,300,220],
    'Fiyat':[2400000,5000000,3500000,6000000,4400000],
    'Oda_Sayisi':[3,5,4,6,4]
    }

df=pd.DataFrame(data)
X=df[['Ev_Buyuklugu','Oda_Sayisi']]
y=df['Fiyat']

# mmodel eğitme
X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# modeli oluştur
model=LinearRegression()
model.fit(X_train,y_train)

# şimdi modelin hata payını kontrol edelim
# y_pred=model.predict(X_test)
# mse=mean_squared_error(y_test,y_pred)
# print(f"Oratlam Kare Hatası: {mse}") # 0.0 çıktı güzel

ev_buyuklugu=float(input("Lütfen Ev Büyüklüğünü m^2 cinsşnden Giriniz: "))
oda_sayisi=int(input("Oda sayısını giriniz:"))
tahmin_fiyat=model.predict([[ev_buyuklugu,oda_sayisi]])
print(f"Evin Tahmini Fiyatı: {tahmin_fiyat[0]:.2f} TL")

