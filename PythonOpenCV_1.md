## Reconnaissance d'objets

### Par l'histogramme

L'histogramme peut être utilisé pour détecter un objet particulier. L'idée est de retrouver l'objet d'intérêt en faisant l'hypothèse que dans l'image, cet objet aura un histrogramme assez proche. Evidemment l'histrogramme peut être différent car, par exemple, l'objet peut être vu d'un autre point de vue, ou il peut être plus petit. Donc l'hypothèse sera d'étudier la similarité entre l'histogramme de l'objet requête et l'histrogramme d'une région de l'image dans laquelle l'objet pourrait être présent. Pour cela nous utilisons la fonction ```cv.CompareHIst(hist_requete,hist_candidat,method)``` où method prend l'une des valeurs suivantes cv2.HISTCMP_CORREL (0), cv2.HISTCMP_CHISQR (1), cv2.HISTCMP_INTERSECT(2) ou cv2.HISTCMP_BHATTACHARYYA (3). Tester les lignes de codes suivantes :
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

```cv.compareHist``` fournit les meilleurs résultats lorsque nous comparons les histogrammes provenant de la même image (heureusement). Ce qui nous permet de vérifier que lorsque la correspondance entre les histogrammes est parfaite les métriques HISTCMP_CORREL et HISTCMP_INTERSECT donnent les valeurs max et pour les deux autres métriques la valeur est minimale. Pourdécider si oui ou non la région de l'image testée comporte l'objet requête, il faut alors comparer la valeur fournie par l'opérateur à un seuil. Ce seuil est à fixer de manière judicieuse afin de limiter le nombre de fausses détections ou de nons détections.
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
