# Introduction en traitement et analyse des images pour des applications de robotique - Les bases

## Mise en place de l'environnement

Python 3, OpenCV, Linux
Python librarie: OpenCV, Numpy, Matplot, Sklearn, Scipy

Tous ces outils ne sont pas nécessairement installés sur vos PC. Par conséquent, les actions suivantes sont à réaliser.
Pour les machines sous Linux, vous devez être sudo sur vos machines. Si ce n'est pas le cas, il vous faudra créer en environnement virtuel dans lequel vous aurez toute liberté d'installer les librairies Python3 que vous allez utiliser. Pour les machines sous Windows, ce problème ne se pose pas nécessairement.

Sous python, l'outil pip permet d'installer les librairies. Cet outil devrait avoir été installé préalablement mais rien n'est moins sûr. Si ce n'est pas le cas, il faudra le faire ainsi (avec les droits sudo ... certains peuvent avoir ces droits et d'autres non) :
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```
Lorsque pip est installé alors il vous faudra installer les modules suivants :

```
sudo pip3 install numpy opencv-contrib-python==3.4.2.4.16 sklearn scipy matplotlib psutil
```

Petit rappel - l'utilisation de ces modules dans vos scripts doit être précédée par les lignes d'import nécessaires à l'usage des
fonctions auxquelles vous souhaitez faire appel. Par exemple :

```
import cv2
import numpy as np
import os
```
## Lecture/Ecriture/affichage d'images

Créer le script suivant qui charge une image de votre disque et l'affiche sur votre écran via le module cv2.
Veillez à renseigner le chemin et nom de l'image que vous souhaitez afficher (ici test_img.png).
```
import imutils
import cv2

# charge unne image dans une variabe définie comme un tableau NumPy multi-dimensionnel 
# donc le shape est nombre rows (height) x nombre columns (width) x nombre channels (depth)
image = cv2.imread("test_img.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# afficher l'image sur l'écran. Attention avec cv2.waitKey(0) vous devrez cliquer dans la fenêtre
# d'affichage et appuyer sur une touche (echap par exemple) pour poursuivre le reste du scirpt
# (ou fermer le script dans le cas présent)
cv2.imshow("Image", image)
cv2.waitKey(0)
```

Pour affichage, préférer l'utilisation de la librairie matplotlib avec les lignes suivantes :
```
from matplotlib import pyplot as plt

img_c = cv.imread('shape_noise.png')
img = cv.cvtColor(img_c, cv.COLOR_BGR2GRAY)

plt.imshow(img_c)
plt.show()
```

## Traitement et analyse couleur

Nous allons désormais faire quelques manipulations du contenu colorimétrique des images que vous aurez à traiter.
Vous savez qu'une image couleur est de base codée en trois canaux RGB et qu'il est possible de la représenter 
dans un autre espace colorimétrique tel que HSV (Teinte/Saturation/Luminance). Toutefois, seule la représentation RGB
peut être afficher sur votre écran.

Vous accèdez aux valeurs de chaque pixel dans chaque canal par les lignes suivantes. Attention, il faut noter qu'OpenCV ne 
représente évidement pas les trois canaux dans l'ordre habituel i.e. RGB mais dans l'ordre BGR. Donc le premier canal (0) est
la composante bleu :
```
blues = image[:, :, 0]
greens = image[:, :, 1]
reds = image[:, :, 2]
```
Dans le cas d'une image HSV, les canaux 0,1 et 2 représentent respectivement les composantes H, S et V de l'image après une conversion.

Pour assurere une telle conversion, vous utiliserez la fonction ```cv2.cvtColor(image, option)``` dans laquelle option peut prendre par exemple l'un
des paramètres suivants :

* cv2.COLOR_BGR2HSV pour une conversion BGR vers HSV
* cv2.COLOR_BGR2GRAY pour une conversion BGR vers niveau de gris
* cv2.COLOR_BGR2YCrCb pour une conversion BGR vers YCrCb
* cv2.COLOR_BGR2HLS pour une conversion BGR vers HLS
* cv2.COLOR_BGR2HLS pour une conversion BGR vers Lab

Les lignes de codes suivantes vous permettent de seuiller les composantes selons certaines valeurs afin de mettre en évidence
que les parties de l'image qui vous intéressent. Dans cet exemple l'image est convertie en HSV (teinte, saturation et luminance) afin d'opérer un seuillage sur la teinte (H).

Dans un premier temps, sont définies les valeurs min et max pour le vert, le rouge et le bleu selon la représentation HSV : c'est
pour cela que seule la première valeur varie...50/60 pour le vert, 170/180 pour le rouge et 110/120 pour le bleu. La teinte prendra une valeur max de 180. La saturation et et la luminance pourront prendre une valeur max de 255.

Puis trois masques sont produits à partir de ces 3 intervalles. Un masque est une image contenant des valeurs 1 ou 0 : un pixel prend la valeur 1 lorsque les valeurs HSV du pixels correspondant est dans l'un des intervalles définis précédemment. Ces trois masques sont finalement utilisés pour effacer les parties de l'image RGB qui ne respectent pas les contraintes colorimétriques imposées.

```
import cv2

# importer la librairie numpy
import numpy as np 

image = cv2.imread('filtering.png')

# cettte image est resizée afin de réduire la quantité de pixels à traiter
image = cv2.resize(image,(300,300))

# changement d'espace colorimétrique
# ici BGR vers HSV : COLOR_BGR2HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# défintion des contraintes min et max sur les composantes ici seule la teinte
# est contrainte
min_green = np.array([50,220,220])
max_green = np.array([60,255,255])

min_red = np.array([170,220,220])
max_red = np.array([180,255,255])

min_blue = np.array([110,220,220])
max_blue = np.array([120,255,255])


# création des masques à partir des limites précédentes
mask_g = cv2.inRange(hsv, min_green, max_green)
mask_r = cv2.inRange(hsv, min_red, max_red)
mask_b = cv2.inRange(hsv, min_blue, max_blue)

# application des masques sur l'image RGB afin de ne garder que les parties qui
#nous intéressent.
res_b = cv2.bitwise_and(image, image, mask= mask_b)
res_g = cv2.bitwise_and(image,image, mask= mask_g)
res_r = cv2.bitwise_and(image,image, mask= mask_r)

# affichage de l'image après sélection de la partie "verte" de l'image
plt.imshow(res_g)
plt.title('Freen')
plt.show()
```

## Binarisation d'une image en niveau de gris

Si nous venons de voir comment mettre en évidence certaines régions d'une image à partir du seuillage colorimétrique de certaines composantes, il est également possible de faire de même sur une image en niveau de gris. Pour cela nous utilisons la fonction ```cv2.threshold()``` de la manière suivante :
```
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_c = cv.imread('shape_noise.png')
img = cv.cvtColor(img_c, cv.COLOR_BGR2GRAY)

ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
```
Des seuillages plus complexes sont possibles offrant généralement de meilleurs résultats dans des contextes où le bruit est plus important (et c'est souivent le cas malheureusement). Deux exemples : le premier un filtrage adaptatif gausssien et un filtrage d'Otsu. 

```
# Binarisation par filtrage adaptatif gaussien
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_c = cv.imread('shape-noise.png')
img = cv.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
	    
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)
	    
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

```

et

```
# Binarisation par méthode d'Otsu
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_c = cv.imread('shape_noise.png')
img = cv.cvtColor(image_c, cv2.COLOR_BGR2GRAY)

# global thresholding
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)

# Otsu's thresholding
ret2,th2 = cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# Otsu's thresholding after Gaussian filtering
blur = cv.GaussianBlur(img,(5,5),0)
ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

# plot all the images and their histograms
images = [img, 0, th1,
          img, 0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
plt.show()
```

## Gestion de la souris et crop d'une image

Voici quelques lignes de codes pour extraire une région d'intérêt à la souris. Grâce à ces quelques lignes il vous sera possible de n'appliquer les lignes précédentes que sur une région de l'image. Mieux encore, cela vous permettra de calculer la valeur moyenne et la variance des composantes d'une partie de l'image, afin de "filtrer" les régions qui lui ressemblent (du point de vue colorimétrique). Dans cet exemple, l'image n'est pas chargée de votre disque dur mais a été acquise via votre webcam.

```
import cv2
import numpy as np

# La ligne qui suit est le point d'entrée lorsque le script python est éxécuté.
# Le script peut être importé comme un module et dans ce cas tout le code placé à la condition du if ne sera pas éxécuté. Toutefois, toutes les fonctions ou classes définies
# avant le if seront accessibles après l'import.
if __name__ == '__main__' :
 
    # initialisation de la webcam
    cap=cv2.VideoCapture(0)
    
    # capture d'une image
    ret, frame=cap.read()
     
    # sélection d'une régions d'intérêt (ROI) à la souris
    r = cv2.selectROI(frame)
    
    # print les informations la région sélectionnée
    print("coin (x,y) = (",r[1],",",r[0],") - taille (dx,dy) = (",r[2],",",r[3],")")
     
    # image croppée (création de la sous-image sélectionnée)
    imCrop = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
 
    # affichage de l'image croppée
    cv2.imshow("Image", imCrop)
    cv2.waitKey(0)
```

Voici quelques lignes de codes pour gérer des actions sur la souris. Elles gères les événements souris tels que le mouvement de la souris (EVENT_MOUSEMOVE), le double click milieu (EVENT_MBUTTONDBLCLK), le click droit (EVENT_RBUTTONDOWN) et le click gauche (EVENT_LBUTTONDOWN). Attention, lorsque vous exécuterez cet exemple, le click droit peut ne pas fonctionner car déjà associé à un menu contextuel. Dans ce cas vous pourrez remplacer cv2.EVENT_RBUTTONDOWN par cv2.EVENT_MBUTTONDOWN.

```
import cv2
import numpy as np

def souris(event, x, y, flags, param):
    global lo, hi, color, hsv_px
    
    if event == cv2.EVENT_MOUSEMOVE:
        # Conversion des trois couleurs RGB sous la souris en HSV
        px = frame[y,x]
        px_array = np.uint8([[px]])
        hsv_px = cv2.cvtColor(px_array,cv2.COLOR_BGR2HSV)
    
    if event==cv2.EVENT_MBUTTONDBLCLK:
        color=image[y, x][0]

    if event==cv2.EVENT_LBUTTONDOWN:
        if color>5:
            color-=1

    if event==cv2.EVENT_RBUTTONDOWN:
        if color<250:
            color+=1
            
    lo[0]=color-5
    hi[0]=color+5

color=100

lo=np.array([color-5, 100, 50])
hi=np.array([color+5, 255,255])

color_info=(0, 0, 255)

cap=cv2.VideoCapture(0)
cv2.namedWindow('Camera')
cv2.setMouseCallback('Camera', souris)
hsv_px = [0,0,0]

while True:
    ret, frame=cap.read()
    image=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(image, lo, hi)
    image2=cv2.bitwise_and(frame, frame, mask= mask)
    cv2.putText(frame, "Couleur: {:d}".format(color), (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, color_info, 1, cv2.LINE_AA)
    
    # Affichage des composantes HSV sous la souris sur l'image
    pixel_hsv = " ".join(str(values) for values in hsv_px)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "px HSV: "+pixel_hsv, (10, 260),
               font, 1, (255, 255, 255), 1, cv2.LINE_AA)
               
    cv2.imshow('Camera', frame)
    cv2.imshow('image2', image2)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Débruitage et post-traitements avec opérateurs morphologiques

Après avoir produite le mask avec ```mask=cv2.inRange(image, lo, hi)``` il est parfois pertinant de débruiter l'image en appliquant un opérteur de lissage par exemple ou en appliquant quelques opérations morphologiques (ouverture, fermeture, erosion, dilatation) sur le mask obtenu.

```
image=cv2.blur(image, (7, 7))
image = cv2.GaussianBlur(image, (11, 11), 0)
mask=cv2.erode(mask, kernel, iterations=4)
mask=cv2.dilate(mask, kernel, iterations=4)
mask=cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask=cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
```

Les opérateurs morphologiques nécessitent la définition d'un noyau (kernel), c'est à dire un voisinage sur lequel l'opérateur sera appliqué.
Lorsque kernel est fixé à None, l'opérateur utilise un kernel par défaut. Voici quelques lignes à tester sur l'image world.png. Jouer avec la taille
du kernel ou la variable iterations (le nombre de fois que l'érosion ou la dilatation est appliquée).

```
import cv2
import numpy as np

img = cv2.imread('world.png',0)
img = cv2.resize(img,(450,450))

# Défintion du kernel de l'érosion : un rectangle de taille 5x5 avec que des 1
kernel = np.ones((5,5),np.uint8)

erosion = cv2.erode(img,kernel,iterations = 1)

# Défintion du kernel de la dilation taille 3x3
kernel = np.ones((3,3),np.uint8)

dilation = cv2.dilate(img,kernel,iterations = 1)

# Défintion du kernel de l'ouverture taille 7x7
kernel = np.ones((7,7),np.uint8)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Défintion du kernel de la fermeture taille 7x7
kernel = np.ones((7,7),np.uint8)

closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original',img)
cv2.imshow('Erosion',erosion)
cv2.imshow('Dilatation',dilatation)
cv2.imshow('Closing',closing)
cv2.imshow('Opening',opening)
cv2.waitKey(0)

```

Ajouter une ou une combinaison de ces 3 lignes dans le script précédent afin de voir leur effet. Vous pourrez jouer sur les différents paramètres afin de mesurer son effet sur le résultat.

### Petit script ++

Voici un petit script appliquant une erosion et une dilatation sur une image en couleur. Cela vous permet de jouer avec la forme du noyau (rectangle, croix ou ellipse) et sa taille.
Modifier ce code pour qu'il prenne en compte une image en niveau de gris. Vous l'appliquerez sur les images __imageasegmenter.jpg__ et __treestosegment.png__ afin de constater l'effet. Ces deux images sont  en couleur donc vous devrez dans un premier temps définir les seuils colorimétriques pour les binariser et ensuite appliquer les erosions et dilatations.

```
import cv2 as cv
import numpy as np
import argparse

src = None

erosion_size = 0

max_elem = 2
max_kernel_size = 21

title_trackbar_element_shape = 'Element:\n 0: Rectangle \n 1: Croix \n 2: Ellipse'
title_trackbar_kernel_size = 'Kernel size:\n 2n +1'
title_erosion_window = 'Erosion Demo'
title_dilation_window = 'Dilation Demo'

def main(image):
    global src
    src = cv.imread(cv.samples.findFile(image))
    if src is None:
        print('Could not open or find the image: ', image)
        exit(0)

    cv.namedWindow(title_erosion_window)
    cv.createTrackbar(title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion)
    cv.createTrackbar(title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion)

    cv.namedWindow(title_dilation_window)
    cv.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    cv.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)

    erosion(0)
    dilatation(0)
    cv.waitKey()


# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv.MORPH_RECT
    elif val == 1:
        return cv.MORPH_CROSS
    elif val == 2:
        return cv.MORPH_ELLIPSE

def erosion(val):
    # Récupère la valeur de la taille du noyau du trackbar de la fenêtre title_erosion_window
    erosion_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window)
    # Récupère le type de noyau désiré (rectangle,croix,ellipse)
    shape_type = cv.getTrackbarPos(title_trackbar_element_shape, title_erosion_window)
    # Récupération du type de shape du noyau
    erosion_shape = morph_shape(shape_type)

    # création du noyau
    element = cv.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
    
    # Application de l'érosion avec l'élément strcturant précédent
    erosion_dst = cv.erode(src, element)
    
    # affichage du résultat
    cv.imshow(title_erosion_window, erosion_dst)

def dilatation(val):
    # Récupère la valeur de la taille du noyau du trackbar de la fenêtre title_dilation_window
    dilatation_size = cv.getTrackbarPos(title_trackbar_kernel_size, title_dilation_window)
    # Récupère le type de noyau désiré (rectangle,croix,ellipse)
    dilatation_shape = cv.getTrackbarPos(title_trackbar_element_shape, title_dilation_window)
    # Récupération du type de shape du noyau
    dilation_shape = morph_shape(dilatation_shape)
    
    # création du noyau
    element = cv.getStructuringElement(dilation_shape, (2 * dilatation_size + 1, 2 * dilatation_size + 1),
                                       (dilatation_size, dilatation_size))
    # Application de la dilatation avec l'élément strcturant précédent
    dilatation_dst = cv.dilate(src, element)
    # affichage du résultat
    cv.imshow(title_dilation_window, dilatation_dst)


if __name__ == "__main__":

    main("./Bureau/fleur.png")
```


## Histogramme d'une image

L'histrogramme représente la distribution des valeurs de tous les pixels de tout ou partie d'une image. OpenCV propose de calculer cet histrogramme avec la fonction cv2.calcHist() de la manière suivante :
```
import cv2
from matplotlib import pyplot as plt
img = cv2.imread('test_img.png')
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(hist,color='b')
plt.show()
```
Pour une image en couleur, [0], [1], [2] indique respectivement que l'histogramme est calculé sur la composante B, G ou R de l'image. None indique qu'aucun masque n'est utilisé. Si un masque est désiré alors il faut le passer en paramètre. [256] indique le nombre de bins utilisé pour calculer l'histogramme. [0,256] indique l'intervalle des valeurs utilisé pour calculer l'histogramme. Ici tout l'intervalle des valeurs est utilisé. L'histogramme est alors affiché avec la fonction plot (ici en bleu color='b').

Pour calculer un histogramme sur une partie d'une image, il suffit de définir un mask et de le passer en paramètre à l'appel de la fonction ```cv2.calcHist()```.
Pour créer un mask vous utiliserez les lignes suivantes :
```
# create a mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[100:300, 100:400] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)
```
Pour comparer, afficher l'image complète et son histogramme puis la région de l'image sélectionnée et son histogramme.


