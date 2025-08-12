from tensorflow.keras.models import load_model

#Se carga el modelo Keras .h5 y se guarda en formato .keras
modelo = load_model("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\implementación_gcp\\modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.h5")
modelo.save("C:\\Users\\Hugo\\Documents\\Info\\Master\\TFM\\Codigos\\implementación_gcp\\modelo_2(no_dropout)_240ts_100e_es38_valloss3.9e-04.keras", save_format="keras")  # Formato moderno

