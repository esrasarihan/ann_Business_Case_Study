#----------------------- Artificial Neural Network for classification --------------------#
# Gerekli kütüphanelerin içe aktarılması
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Tensorflow sürümünün kontrol edilmesi
print(tf.__version__)

# Verinin yüklenmesi
bank_data = pd.read_csv("Artificial_Neural_Network_Case_Study_data.csv")

# Bağımsız değişkenleri (özellik matrisi) alırken tüm satırları ve son sütunu alıyoruz
# Satır numaraları ve müşteri kimlikleri modellemede gerekli değil, bu nedenle bunları atıyoruz ve kredi skoru ile başlıyoruz
X = bank_data.iloc[:, 3:-1].values
print("Independent variables are:", X)
# Bağımlı değişkeni alırken tüm satırları ve yalnızca son sütunu alıyoruz
y = bank_data.iloc[:, -1].values
print("Dependent variable is:", y)

# Cinsiyet değişkeninin dönüştürülmesi, etiketler rastgele seçilir
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
print(X)

# Coğrafya sütun değişkeninin dönüştürülmesi, etiketler rastgele seçilir, ct [1] argümanı hedef sütunun dizinini istiyor
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

# Veriyi eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# bu anlık fotoğrafların her birinin satır ve sütunların boyutlarını görmek için boyutlarını yazdırmak
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

# NN'ye gidecek özelliklerin Veri Ölçeklendirmesi/normalleştirilmesi
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#----------------------- Modelin Oluşturulması -----------------------#

# Yapay Sinir Ağı'nın başlatılması, Keras'tan Tensorflow'un Sıralı sınıfını çağırarak
ann = tf.keras.models.Sequential()

# Sıralı Ağa "tam bağlı" GİRİŞ katmanının eklenmesi, Keras'tan Dense sınıfını çağırarak
# Birim Sayısı = 6 ve Aktivasyon Fonksiyonu = Doğrultucu (Rectifier)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Sıralı Ağa "tam bağlı" İKİNCİ katmanın eklenmesi, Keras'tan Dense sınıfını çağırarak
# Birim Sayısı = 6 ve Aktivasyon Fonksiyonu = Doğrultucu (Rectifier)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# Sıralı Ağa "tam bağlı" ÇIKTI katmanının eklenmesi, Keras'tan Dense sınıfını çağırarak
# Birim Sayısı = 1 ve Aktivasyon Fonksiyonu = Sigmoid
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#----------------------- Modelin Eğitilmesi -----------------------#
# Ağın Derlenmesi
# Optimizasyon Türü = Adam Optimizasyonu, Kayıp Fonksiyonu = ikili bağımlı değişken için çaprazentropi, ve Optimizasyon doğruluğa göre yapılır
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modelin Eğitilmesi
# batch_size = 32, varsayılan değer, epoch sayısı = 100
ann.fit(X_train, y_train, batch_size=32, epochs=100)

#----------------------- Modelin Değerlendirilmesi ---------------------#
# Amacımız bu YSA modelini müşterinin bankayı terk etme olasılığını tahmin etmek için kullanmak
# Tek bir gözlem için terk olasılığını tahmin etme

# Coğrafya: Fransızca
# Kredi Skoru:600
# Cinsiyet: Erkek
# Yaş: 40 yaşında
# Tenure: 3 yıl
# Bakiye: $60000
# Ürün Sayısı: 2
# Kredi Kartı Var
# Aktif üye
# Tahmini Maaş: $50000

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# Bu müşterinin bankayı terk etme olasılığı %4'tür

# tahminlerin ve gerçek değerlerin vektörünü gösterme
# olasılıklar
y_pred_prob = ann.predict(X_test)

# olasılıkları ikiliye dönüştürme
y_pred = (y_pred_prob > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

# Karmaşıklık Matrisi
confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix", confusion_matrix)
print("Accuracy Score", accuracy_score(y_test, y_pred))

# Karmaşıklık Matrisinin Görselleştirilmesi
sns.set(font_scale=1.4)   # etiket boyutu için
plt.figure(figsize=(6, 4))  # gerekiyorsa şekil boyutunu ayarlayın
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# Yeni veriler üzerinde tahmin yapma
new_data = [[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000],
            [0, 0, 1, 700, 0, 35, 5, 80000, 1, 0, 1, 70000],
            [0, 1, 0, 550, 1, 45, 7, 90000, 3, 1, 0, 60000]]

# Yeni verileri ölçeklendirme
scaled_new_data = sc.transform(new_data)

# Tahminlerde Bulunma
predictions = ann.predict(scaled_new_data)
predictions_binary = (predictions > 0.5).flatten()

# Tahminleri Gösterme
for i, pred in enumerate(predictions_binary):
    print(f"Prediction for data {i+1}: {'Leave' if pred else 'Stay'}")

print("------ Detailed Analysis ------")
# Sınıflandırma Raporu
report = classification_report(y_test, y_pred, target_names=["Stay", "Leave"])
print(report)

# Sınıflandırma raporunu işleme
labels = ['Stay', 'Leave']
precision = [0.88, 0.75]
recall = [0.96, 0.50]
f1_score = [0.92, 0.60]
support = [1595, 405]

# Sınıflandırma raporunu histogramlarla görselleştirme
metrics = ['Precision', 'Recall', 'F1-Score', 'Support']
values = [precision, recall, f1_score, support]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, metric, value in zip(axes, metrics, values):
    ax.bar(labels, value, color=['blue', 'orange'])
    ax.set_title(metric)
    ax.set_ylim(0, 1.1 if metric != 'Support' else max(support) * 1.1)
    for i in range(len(value)):
        ax.text(i, value[i] + 0.05 if metric != 'Support' else value[i] + max(support) * 0.05, f'{value[i]:.2f}' if metric != 'Support' else f'{value[i]}', ha='center')

plt.tight_layout()
plt.show()

# -------------------------------- Regression Part ----------------------------------#
# Modeli oluşturma
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred_reg = regressor.predict(X_test)

# Performans metriklerini hesaplama
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Kullanıcı verilerini tanımlama
user_data = {
    "esra": [1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000],
    "melis": [0, 0, 1, 700, 0, 35, 5, 80000, 1, 0, 1, 70000],
    "berke": [0, 1, 0, 550, 1, 45, 7, 90000, 3, 1, 0, 60000],
    "hilal": [1, 0, 0, 620, 1, 30, 2, 50000, 1, 1, 1, 45000]
}

user_names = list(user_data.keys())
user_values = list(user_data.values())

# Kullanıcı verilerini ölçeklendirme
scaled_user_data = sc.transform(user_values)

# Kullanıcı verileri üzerinde tahminlerde bulunma
user_predictions = ann.predict(scaled_user_data)
user_predictions_binary = (user_predictions > 0.5).flatten()

# Tahmin sonuçlarını görselleştirme
prediction_labels = ['Stay' if not pred else 'Leave' for pred in user_predictions_binary]

plt.figure(figsize=(10, 6))
bars = plt.bar(user_names, user_predictions.flatten(), color=['blue' if label == 'Stay' else 'red' for label in prediction_labels])
plt.xlabel('Users')
plt.ylabel('Churn Probability')
plt.title('Bank Churn Probability for Users')
plt.ylim(0, 1)

for bar, label in zip(bars, prediction_labels):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, label, ha='center', va='bottom')

plt.show()

# Kullanıcı tahminlerini yazdırma
for name, pred in zip(user_names, user_predictions.flatten()):
    print(f"{name.capitalize()} için bankadan ayrılma olasılığı: {pred:.2f}")
