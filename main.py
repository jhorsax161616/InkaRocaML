# Librerias
from tkinter import *
from PIL import Image, ImageTk
import imutils
import cv2
import numpy as np
from ultralytics import YOLO
import math


def run() -> None:
    # Declarando variables globales
    global cap, seccion_video, model, nombres_clases, img_background, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf, pantalla
    
    # Ejecutamos Ventana principal
    ventana_principal()
    
    # Cargando el modelo YOLO
    model = YOLO("/models/best110.pt")

    # Clases: 0 -> llavero | 1 -> Chompa | 2 -> Guantes | 3 -> Gorro
    nombres_clases = ["Llavero", "Chompa", "Guantes", "Gorro"]

    # Cargando imagenes de productos con OpenCV
    img_llavero = cv2.imread("./img/interface/llavero.png")
    img_chompa = cv2.imread("./img/interface/chompa.png")
    img_guantes = cv2.imread("./img/interface/guantes.png")
    img_gorro = cv2.imread("./img/interface/gorro.png")

    # Cargando imagenes de información de productos con OpenCV
    img_llavero_inf = cv2.imread("./img/interface/llavero_inf.png")
    img_chompa_inf = cv2.imread("./img/interface/chompa_inf.png")
    img_guantes_inf = cv2.imread("./img/interface/guantes_inf.png")
    img_gorro_inf = cv2.imread("./img/interface/gorro_inf.png")

    # Iniciando la captura de video
    iniciar_video_camara()

    # Loop de la ventana
    pantalla.mainloop()


def ventana_principal() -> None:
    # Inicializando ventana principal
    global pantalla, img_background
    pantalla = Tk()
    pantalla.title("INKAROCA ML")
    pantalla.geometry("1920x1080")

    # Background
    img_background = ImageTk.PhotoImage(Image.open("./img/interface/background.png"))
    background = Label(image=img_background, text="INKAROCA ML")
    background.place(x=0, y=0, relwidth=1, relheight=1)

def iniciar_video_camara() -> None:
    global cap, seccion_video, pantalla

    # Inicializando la captura de video
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 914)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 665)

    # Sección de videocamara en la ventana principal (521, 314)
    seccion_video = Label(pantalla)
    seccion_video.place(x=521, y=314)

    # Actualizando la sección de video
    actualizar_video()

def actualizar_video() -> None:
    pass


if __name__ == "__main__":
    run()