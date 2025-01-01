import joblib
import os

import warnings
warnings.filterwarnings("ignore")

def manual_rf_prediction(ph, temperature, turbidity):
    try:
        # Model dosyasının varlığını kontrol et
        model_path = "veri_modeli_son.pkl"
        if not os.path.exists(model_path):
            print("Model dosyasi bulunamadi.")
            return

        # Modeli yükle
        rf_model = joblib.load(model_path)
        print("Model basariyla yuklendi.")
        
        # Giriş verilerini bir listeye dönüştür
        input_data = [[ph, turbidity, temperature]]
        
        # Tahmin yap
        prediction = rf_model.predict(input_data)
        
        # Tahmin sonucunu kullanıcıya göster
        print(f"Tahmin edilen balik turu: {prediction[0]}")
    except Exception as e:
        print(f"Bir hata olustu: {e}")

# Kullanıcidan giris al
try:
    print("Balik turu tahmini icin lutfen asagidaki bilgileri giriniz.")
    ph = float(input("pH degerini girin (pH): "))
    temperature = float(input("Bulanikli degerini girin (NTU): "))
    turbidity = float(input("Sicaklik degerini girin (°C): "))
    manual_rf_prediction(ph, turbidity, temperature)
except ValueError:
    print("Gecerli bir sayi girin.")