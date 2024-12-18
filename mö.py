import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# veriyi hazırlama
# önce basit bir data oluşturduk exceden de çekebşlşrşz

data={
    'Ev_Buyuklugu':[120,250,175,300,220],
    'Fiyat':[2400000,5000000,3500000,6000000,4400000]
}

df=pd.DataFrame(data)   #veriyi df ye çevirme

X=df[['Ev_Buyuklugu']] #burara [[]] 2 tane kareli parantez olma sebebi bu girdi olur ve başlık ile brebaer değerleri alır girdileri 
y=df['Fiyat'] #bu ise çıktıdır burada sadece değerlere ihtiyacımız var ve [] tek paranteez de bize çıktıları yani değerleri veriri başlığı vermez
# X_tarin(x'i eğitme anlamı) ve x_tste(x'itest etme anlamı aslında) test_size(%20 eğitcez anlamı) random_state(anlamı yok gibi)
X_train , X_test , y_train , y_test=train_test_split(X,y,test_size=0.2, random_state=42)


# Modeli oluştur 
model=LinearRegression()
model.fit(X_train,y_train)

# biz burada modelin hatasının ne kadar olduğunu ölçüyoruz ve hatamız 0.0 çıktı şimdi bunu yorum satırına alabilrz
# y_pred=model.predict(X_test)
# HATA NE KADAR KÜÇÜKSE TAHMİN O KADAR İYİDİR
# mse=mean_squared_error(y_test,y_pred)
# print(f"Oratlam Kare Hatası: {mse}")

ev_buyuklugu=float(input("Lütfen Ev Büyüklüğünü m^2 cinsşnden Giriniz: "))
tahmini_fiyat=model.predict([[ev_buyuklugu]])
print(f"Evin Tahmini Fiyatı:{tahmini_fiyat[0]:.2f}TL")