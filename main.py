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
    global cap, lblVideo, model, nombres_clases, img_background, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf, pantalla
    
    # Ejecutamos Ventana principal
    ventana_principal()

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


if __name__ == "__main__":
    run()