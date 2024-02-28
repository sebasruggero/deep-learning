import subprocess

# Instalar las librerías necesarias
subprocess.run(['pip', 'install', 'gradio', 'fastai'])

# Ahora importa las librerías recién instaladas
from fastai.vision.all import *
import gradio as gr

# El resto de tu script sigue aquí
learn = load_learner('model/osos_model.pkl')

categories = [
    "Andean Bear or Spectacled Bear",
    "Asiatic Black Bear",
    "Brown Bear",
    "Giant Panda",
    "North American Black Bear",
    "Polar Bear",
    "Sloth Bear",
    "Sun Bear",
    "Teddy"
]

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

demo = gr.Interface(fn=classify_image, inputs="image", outputs="label")
demo.launch()
