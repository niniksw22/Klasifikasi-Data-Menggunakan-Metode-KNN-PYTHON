import pandas as pd
import numpy as np

cryotherapy = pd.read_csv('cryotherapy.csv', engine='python')
cryotherapy.head()
cryotherapy.info()

# Variabel independen
x = cryotherapy.drop(["Result_of_Treatment"], axis = 1)
x.head()

# Variabel dependen
y = cryotherapy["Result_of_Treatment"]
y.head()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier

# Mengaktifkan fungsi klasifikasi
klasifikasi = KNeighborsClassifier(n_neighbors=5)

# Memasukkan data training pada fungsi klasifikasi
klasifikasi.fit(x_train, y_train)

# Menentukan hasil prediksi dari x_test
y_pred = klasifikasi.predict(x_test)

# Menentukan probabilitas hasil prediksi
klasifikasi.predict_proba(x_test)

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#nilai akurasi
from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred)