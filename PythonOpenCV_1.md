
# Introduction en traitement et analyse des images pour des applications de robotique (Partie 1)

## Mise en place de l'environnement

Python 3, OpenCV, Linux
Python librarie: OpenCV, Numpy, Matplot, Sklearn, Scipy

Tous ces outils ne sont pas nécessairement installés sur vos PC. Par conséquent, les actions suivantes sont à réaliser.
Vous devez être sudo sur vos machines. Si ce n'est pas le cas, il vous faudra créer en environnement virtuel dans lequel vous aurez toute liberté d'installer les librairies Python3 que vous allez utiliser.

Sous python, l'outil pip permet d'installer les librairies. Cet outil devrait avoir été installé préalablement mais rien n'est moins sûr. Si ce n'est pas le cas, il faudra le faire ainsi (avec les droits sudo ... certains peuvent avoir ces droits et d'autres non) :
```
wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py
```

Lorsque pip est installé alors il vous faudra installer les modules suivants :

```
sudo pip3 install --proxy=http://10.100.1.4:8080 numpy tensorflow opencv-contrib-python==3.4.2.16 sklearn scipy matplotlib psutil
```
Petit rappel - l'utilisation de ces modules dans vos scripts est réalisé par exemple ainsi :
```
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
```
Si tout a été installé convenablement, alors chacune des lignes précédentes devraient n'engendrer aucune erreur.

## Les bases

### Lecture/Ecriture/affichage d'images

Créer le script suivant qui charge une image de votre disque et l'affiche sur votre écran via le module cv2.
Veillez à renseigner le chemin et nom de l'image que vous souhaitez afficher (ici .png).
```
import imutils
import cv2

# charge unne image dans une variabe définie comme un tableau NumPy multi-dimensionnel 
# donc le shape est nombre rows (height) x nombre columns (width) x nombre channels (depth)
image = cv2.imread("jp.png")
(h, w, d) = image.shape
print("width={}, height={}, depth={}".format(w, h, d))

# afficher l'image sur l'écran. Attention avec cv2.waitKey(0) vous devrez cliquer dans la fenêtre
# d'affichage et appuyer sur une touche (echap par exemple) pour poursuivre le reste du scirpt
# (ou fermer le script dans le cas présent)
cv2.imshow("Image", image)
cv2.waitKey(0)
```

A partir du script d'acquisition que M. Boonaert vous a remis et notamment de la fonction ```GrabImageFromCam```
créer la fonction ```DisplayImage()``` permettant d'afficher dans une fenêtre OpenCV l'image que vous venez d'acquérir.
Ajouter cette fonction dans un nouveau script que vous nommerez ```ComputerVisionNom1Nom2.py```
(Nom1 et Nom2 sont les noms des étudiants de votre groupe)
Ce script pourra contenir toutes les fonctions que vous aurez développées pour le projet de cette UV.

### Tratement et analyse couleur

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
Les lignes de codes suivantes vous permettent de seuiller les composantes selons certaines valeur afin de mettre en évidence
que les parties de l'image qui vous intéressent. Dans cet exemple l'image est convertie en HSV. Dans un premier temps, sont définies
les valeurs min et max pour le vert, le rouge et le bleu selon la représentation HSV : c'est pour cela que seule la première valeur
varie...50/60 pour le vert, 170/180 pour le rouge et 110/120 pour le bleu. Puis trois masques sont produits à partir de ces 3 intervalles :
un masque est une image contenant des valeurs 1 ou 0 : un pixel prend la valeur 1 lorsque les valeurs HSV du pixels correspondant
est dans l'un des intervalles définis précédemment. Ces trois masques sont finalement utilisés pour effacer les parties de l'image RGB
qui ne respectent pas les contraintes colorimétriques imposées.

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
cv2.imshow('Green',res_g)
```

### Gestion de la souris et crop d'une image

Voici quelques lignes de codes pour extraire une région d'intérêt à la souris. Grâce à ces quelques lignes il vous sera possible de n'appliquer les lignes précédentes que sur une région de l'image. Mieux encore, cela vous permettra de calculer la valeur moyenne et la variance des composantes d'une partie de l'image, afin de "filtrer" les régions qui lui ressemblent (du point de vue colorimétrique). Dans cet exemple, l'image n'est pas chargée de votre disque dur mais a été acquise via votre webcam.

```
import cv2
import numpy as np
 
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

Après avoir produite le mask avec ```mask=cv2.inRange(image, lo, hi)``` il est parfois pertinant de débruiter l'image résultats en lissant ou par quelques opérations morphologiques (ouverture, fermeture, erosion, dilatation).
```
image=cv2.blur(image, (7, 7))
image = cv2.GaussianBlur(image, (11, 11), 0)
mask=cv2.erode(mask, None, iterations=4)
mask=cv2.dilate(mask, None, iterations=4)
```

Ajouter une ou une ombinaison de ces 3 lignes dans le script précédent afin de voir leur effet. Vous pourrez jouer sur les différents paramètres afin de mesurer son effet sur le résultat.

### Histogramme d'une image

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

## Reconnaissance d'objets
### Par l'histogramme
L'histogramme peut être utilisé pour détecter un objet particulier. Pour cela nous utilisons la fonction ```cv.CompareHIst(hist_requete,hist_candidat,method)``` où method prend l'une des valeurs suivantes cv2.HISTCMP_CORREL (0), cv2.HISTCMP_CHISQR (1), cv2.HISTCMP_INTERSECT(2) ou cv2.HISTCMP_BHATTACHARYYA (3). Tester les lignes de codes suivantes :
```
from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np

src_base = cv.imread("main1.jpg")
src_test1 = cv.imread("main2.jpg")
src_test2 = cv.imread("main3.jpg")

hsv_base = cv.cvtColor(src_base, cv.COLOR_BGR2HSV)
hsv_test1 = cv.cvtColor(src_test1, cv.COLOR_BGR2HSV)
hsv_test2 = cv.cvtColor(src_test2, cv.COLOR_BGR2HSV)

hsv_half_down = hsv_base[hsv_base.shape[0]//2:,:]
h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]

# Hue varie de 0 à 179, la saturation varie de 0 to 255
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges # concat lists

# Utilise les canaux 0 et 1 pour calculer l'histogramme (H et S)
channels = [0, 1]

hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges)
cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_half_down = cv.calcHist([hsv_half_down], channels, None, histSize, ranges)
cv.normalize(hist_half_down, hist_half_down, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_test1 = cv.calcHist([hsv_test1], channels, None, histSize, ranges)
cv.normalize(hist_test1, hist_test1, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

hist_test2 = cv.calcHist([hsv_test2], channels, None, histSize, ranges)
cv.normalize(hist_test2, hist_test2, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

for compare_method in range(4):
    base_base = cv.compareHist(hist_base, hist_base, compare_method)
    base_half = cv.compareHist(hist_base, hist_half_down, compare_method)
    base_test1 = cv.compareHist(hist_base, hist_test1, compare_method)
    base_test2 = cv.compareHist(hist_base, hist_test2, compare_method)
    
    # affiche les résultats de l'opérateur de comparaison
    print('Method:', compare_method, 'Perfect, Base-Half, Base-Test(1), Base-Test(2) :',\
          base_base, '/', base_half, '/', base_test1, '/', base_test2)
```

```cv.compareHist``` fournit les meilleurs résultats lorsque nous comparons les histogrammes provenant de la même image (heureusement). Ce qui nous permet de vérifier que lorsque la correspondance entre les histogrammes est parfaite les métriques HISTCMP_CORREL et HISTCMP_INTERSECT donnent les valeurs max et pour les deux autres métriques la valeur est minimale.
Cette méthode est pertinente lorsque vous avez déjà un ensemble de régions candidates dans une image pour lesquelles vous souhaitez savoir si elles correspondent à l'objet recherché (hist_requete).

### Par template matching 
Une autre manière d'opérer la reconnaissance et d'utiliser la fonction ```cv2.matchTemplate``` qui recherche le template candidat i.e. l'image de l'objet que nous recherchons en la faisant "glisser" sur toute l'image. Voici quelques lignes de codes à tester cette fois sur une image en niveau de gris :

```
import cv2
import numpy as np

# chargement d'une image
img_rgb = cv2.imread('roadsign.png')

# conversio en niveau de gris (un seul canal)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# chargement de l'image template à rechercher
template = cv2.imread('sign_stop.png',0)
w, h = template.shape[::-1]

# seuil de décision qui valide ou non le matching
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where( res >= threshold)

# affiche tous les matchs validés sur l'image originale
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
    
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
### Synthèse

Ecrire un script capable de retrouver dans le flux de votre webcam une région de l'image que vous aurez préalablement sélectionnée à l'aide de votre souris.
Il pourra s'agir d'un élément de votre visage par exemple (votre oeil, votre bouche ...).
Vous utiliserez la fonction ```cv2.matchTemplate```. A chaque nouvelle image acquise, les régions candidates seront localisées grâce à un rectangle.
Si votre caméra ne fonctionne pas vous utiliserez la vidéo chris2.mp4 et vous tenterez de retrouver un des éléments du visage de Chris.
Pour ouvrir le flux d'une vidéo vous utiliserez les lignes de codes suivantes :

```
cap = cv2.VideoCapture('video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False):
	  print("Error opening video stream or file")
   
# Read until video is completed
while(cap.isOpened()):
   # Capture frame-by-frame
	  ret, frame = cap.read()
	  if ret == True:
       ... votre code ...
   else:
       break
cap.release()
```
