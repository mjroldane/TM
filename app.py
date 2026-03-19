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

# --- Carga del Modelo (Cache para mayor velocidad) ---
@st.cache_resource
def load_my_model():
    # Asegúrate de que el archivo 'keras_model.h5' esté en la misma carpeta
    return load_model('keras_model.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")

# --- Interfaz Lateral ---
with st.sidebar:
    st.header("Configuración")
    st.info("Este modelo identifica a Camilo y María José usando Visión Artificial.")
    # Imagen decorativa si la tienes, si no, puedes comentar esta línea
    # st.image('OIG5.jpg', width=200)

# --- Entrada de Cámara ---
img_file_buffer = st.camera_input("Toma una foto para iniciar el saludo")

if img_file_buffer is not None:
    # 1. Procesamiento de la imagen
    img = Image.open(img_file_buffer).convert("RGB")
    size = (224, 224)
    # ImageOps.fit recorta y ajusta la imagen al tamaño necesario para el modelo
    img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    
    # 2. Convertir a Array y Normalizar
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    
    # Crear el contenedor para el modelo (batch size 1, 224x224, 3 canales RGB)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 3. Predicción
    with st.spinner("Identificando..."):
        prediction = model.predict(data)
        index = np.argmax(prediction) # Índice con mayor probabilidad
        confidence_score = prediction[0][index]

    # 4. Lógica de Saludos Personalizados
    # AJUSTE IMPORTANTE: 
    # Index 0 suele ser la primera clase que creaste en Teachable Machine
    # Index 1 suele ser la segunda clase.
    
    st.divider() # Línea visual para separar el resultado

    if confidence_score > 0.80: # Umbral de confianza del 80%
        if index == 0:
            st.balloons()
            st.header("¡Hola Camilo! 👋")
            st.write(f"Nivel de confianza: {confidence_score:.2%}")
            
        elif index == 1:
            st.snow() # Un efecto visual lindo para María José
            st.markdown("<h1 style='text-align: center; color: #E91E63;'>¡Hola María José! ✨</h1>", unsafe_allow_html=True)
            st.markdown("<h3 style='text-align: center;'>Estás hermosa hoy.</h3>", unsafe_allow_html=True)
            st.write(f"Nivel de confianza: {confidence_score:.2%}")
        else:
            st.subheader("Persona detectada pero no tiene saludo asignado.")
    else:
        st.warning("La confianza es muy baja. Por favor, intenta iluminar mejor tu rostro.")

else:
    st.write("Esperando captura de cámara...")
