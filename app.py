import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import platform

# --- Configuración de la Página ---
st.set_page_config(page_title="Reconocimiento Facial", page_icon="✨")

st.title("Sistema de Reconocimiento")
st.write(f"Ejecutando en Python: {platform.python_version()}")

# --- Carga del Modelo ---
@st.cache_resource
def load_my_model():
    return load_model('keras_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- Interfaz Lateral ---
with st.sidebar:
    st.header("Configuración")
    st.info("Intercambiamos los nombres para que el saludo sea correcto.")

# --- Entrada de Cámara ---
img_file_buffer = st.camera_input("Toma una foto para iniciar el saludo")

if img_file_buffer is not None:
    # 1. Procesamiento de la imagen
    img = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    
    # 2. Convertir a Array y Normalizar
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 3. Predicción
    with st.spinner("Identificando..."):
        prediction = model.predict(data)
        index = np.argmax(prediction) 
        confidence_score = prediction[0][index]

    st.divider() 

    # 4. Lógica de Saludos CORREGIDA
    if confidence_score > 0.80:
        # INTERCAMBIAMOS AQUÍ: Antes el 0 era Camilo, ahora es María José (o viceversa según tu modelo)
        if index == 0:
            st.snow() 
            st.markdown("<h1 style='text-align: center; color: #E91E63;'>¡Hola María José! ✨</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Estás hermosa hoy.</h3>", unsafe_allow_html=True)
            st.write(f"Nivel de confianza: {confidence_score:.2%}")
            
        elif index == 1:
            st.balloons()
            st.header("¡Hola Camilo! 👋")
            st.write(f"Nivel de confianza: {confidence_score:.2%}")
        else:
            st.subheader("Persona detectada pero sin saludo asignado.")
    else:
        st.warning("Confianza baja. ¡Ajusta la luz o acércate más!")

else:
    st.write("Esperando captura de cámara...")
