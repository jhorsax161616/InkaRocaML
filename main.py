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
    model = YOLO("./models/best110.pt")

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

    # Refrescar el video cada 10ms
    refrescar_video()

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
    # Sección de videocamara en la ventana principal (521, 314)
    seccion_video = Label(pantalla)
    seccion_video.place(x=521, y=314)

    # Inicializando la captura de video
    cap = cv2.VideoCapture(1)
    # Si no funciona para windows, usar
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # Resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 914)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 665)


def refrescar_video() -> None:
    global cap, seccion_video, model, nombres_clases, img_background, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf, pantalla

    # Leyendo el frame de la cámara si es valido
    if cap:
        ret, frame = cap.read()
        # Procesando el frame
        frame = cv2.flip(frame, 1) # Efecto espejo
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if ret:
            # Obteniendo datos del objeto detectado
            try:
                x1, y1, x2, y2, clase, confidencia =  obteniendo_datos_del_objeto(frame)
            except TypeError:
                x1, y1, x2, y2, clase, confidencia = None, None, None, None, None, 0

            # Dibujando la caja del objeto detectado, si la confidencia es mayor a 50%
            if confidencia > 50:
                dibujar_caja_del_objeto(x1, y1, x2, y2, clase, confidencia, frame)

            # Redimensionar el frame
            frame = imutils.resize(frame, width=914)

            # Convertir el video
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)

            # Mostrando el video en la ventana
            seccion_video.configure(image=img)
            seccion_video.image = img
            seccion_video.after(10, refrescar_video) # Actualizar cada 10ms

        else:
            cap.release() # Liberar la cámara


def obteniendo_datos_del_objeto(frame) -> tuple:
    global cap, model, nombres_clases, img_llavero, img_chompa, img_guantes, img_gorro
    global img_llavero_inf, img_chompa_inf, img_guantes_inf, img_gorro_inf

    # Obteniendo los datos del objeto con el modelo YOLO
    results = model(frame, stream=True, verbose=False)

    # Recorriendo los resultados
    for res in results:
        # Extrayendo las cajas
        cajas = res.boxes
        # Recorriendo las cajas
        for caja in cajas:
            # Obteniendo las coordenadas de la caja en enteros
            x1, y1, x2, y2 = [int(val) for val in caja.xyxy[0]]

            # Corrigiendo error para objetos al limite de la captura
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            if x2 > 914: x2 = 914
            if y2 > 665: y2 = 665

            # Obteniendo el nombre de la clase por el indice
            clase: str = nombres_clases[int(caja.cls[0])]

            # Obteniendo la confidencia en porcentaje
            confidencia: int = math.ceil(caja.conf[0] * 100)
            
            return x1, y1, x2, y2, clase, confidencia
        
def dibujar_caja_del_objeto(x1, y1, x2, y2, clase, confidencia, frame) -> None:
    # Clasificando el objeto
    match clase:
        case "Llavero":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (237, 64, 61), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando el llavero
            dibujar_producto()
        case "Chompa":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (253, 154, 143), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando la chompa
            dibujar_producto()
        case "Guantes":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (252, 113, 34), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando los guantes
            dibujar_producto()
        case "Gorro":
            # Dibujando la caja del objeto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (242, 176, 34), 2)
            # Dibujando el nombre de la clase y la confidencia
            cv2.putText(frame, f"{clase} {confidencia}%", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            # Dibujando el gorro
            dibujar_producto()


def dibujar_producto() -> None:
    pass
    

if __name__ == "__main__":
    run()