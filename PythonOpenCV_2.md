# Introduction en traitement et analyse des images pour des applications de robotique - Détection d'objets

## Détection d'ensemble de pixels connexes

Dans le TP précédent, vous avez appris à sélectionner certaines parties d'une image à partir de l'analyse de ses composantes colorimétriques ou plus simplement de ses niveaux de gris (```fonction cv2.inRange()``` et  ```cv2.threshold()```.
Ces deux fonctions produisent des masques dans lesquels les pixels à 255 respectent la ou les contraintes imposées. Il est alors possible de détecter les éléments connexes dans le mask et d'en extraire certaines informations de forme ou de position (```cv2.findcontours()```). En considérant que vous avez préalablement  extrait un **mask** d'une image que vous avez chargée dans la variable **im**, voici les lignes de code qui vous permettent d'extraire tous les éléments connexes, de classer les différents ensembles de pixels connexes par ordre croissant de leur surface (fonction ```sorted()``` avec paramètre ***key=cv.contourArea***), d'extraire le cercle minimum qui contient le 1600ème ensemble de pixels connexes pour finalement l'afficher dans la copy de l'image initiale (2 est ici la taille du trait avec lequel le cercle est dessiné dans l'image). Pour plus d'informations, vous pouvez vous référer à https://docs.opencv.org/4.5.0/dd/d49/tutorial_py_contour_features.html (pour la version 4.5 d'OpenCV).

```
image=im.copy()

# vérifier la version d'OpenCV que vous avez installée
# print(cv2.__version__)

# pour une version 3.x d'OpenCV
image2,elements,hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# pour la version 4.x d'OpenCV
#elements,hierarchy=cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# len(elementsà renvoie le nombre de régions de pixels connexes détectés.
if len(elements) > 0:

    # sorted() permet de trier par ordre décroissant les instances de elements en fonction de leur surface
    c=sorted(elements, key=cv2.contourArea)

    # défini le cercle minimum qui couvre complètement l'objet avec une surface minimale
    ((x, y), rayon)=cv2.minEnclosingCircle(c[1600])

    # Affichage
    cv2.circle(image, (int(x), int(y)), int(rayon), [0,0,255], 2)
    cv2.putText(image, "Objet !!!", (int(x)+10, int(y) -10), cv.FONT_HERSHEY_DUPLEX, 1, [0,255,0], 1, cv.LINE_AA)
```
Tester ces quelques sur l'image ***imageasegmenter.jpg*** et ***treestosegment.png***.

Comparer ces résultats avec ceux obtenus après avoir appliquer des opérations morphologiques. Par exemple, pour supprimer les petits ensembles vous pouvez appliquer des ouvertures ou une succession de plusieurs érosions et autant de dilatations (pour récupérer la taille initiale des régions que vous souhaitez garder).

Il vous est possible d'afficher tout ou partie des ensembles de pixels connexes en utilisant la fonction ```cv.drawContours()```. Par exemple pour les  afficher en couleur vert (0,255,0) dans la variable ***image*** il suffit d'utiliser ```cv.drawContours(image, elements, -1, (0,255,0), 3)```. Si vous souhaitez afficher le 1600ème ensemble de pixels connexes cette fois en rouge (0,0,255) : 
```
# La région que vous souhaitez afficher
cnt = elements[1600]
cv.drawContours(img, [cnt], 0, (0,0,255), 3)
```

Il peut être également très utilie de produire le mask correspondant à l'un des ensembles de pixels connexes extraits :

```
mask = np.zeros(im.shape,np.uint8)
cv.drawContours(mask,[elements[1600]],0,255,-1)
```
Une fois que vous avez extrait tous les ensembles de pixels connexes dans l'image binarisée, il peut s'avérer très utile d'en extraire des caractéristiques de forme afin de déterminer par exemple un classifieur d'objets :

* Rapport d'aspect
```
x,y,w,h = cv.boundingRect(c[1600])
aspect_ratio = float(w)/h
```

* Valeur max, valeur min et leur position dans l'image en utilisant le mask d'un ensembe de pixels connexes
```
min_val, max_val, min_loc, max_loc = cv.minMaxLoc(image,mask = mask)
```

* Orientation du grand axe de l'ellipse qui approxime la forme d'un ensemble de pixels
```
(x,y),(MA,ma),angle = cv.fitEllipse(c[1600])
```

* Couleur moyenne et intensité moyenne
```
mean_val = cv.mean(im,mask = mask)
```

Tous ces paramètres sont utilies pour caractériser la forme représentée par un ensemble de pixels connexes. Un ensemble de ces paramètres peut être calculé et être regroupé dans un vecteur qui caractérisera la forme en question. Ainsi toute forme "segmentée" peut être résumée par un vecteur dans l'espace de ces caractéristiques.  L'idée sous-jacente est de trouver des sous-régions de cet espace qui regrouperaient les ensembles de pixels connexes dont la géométrie serait similaire (donc appartenant hypothétiquement à une même classe)

D'autres caractéristiques existent. MATLAB en proposent d'autres que vous pourriez implanter et utiliser à l'avenir : http://www.mathworks.in/help/images/ref/regionprops.html

Le module skimage propose un ensemble plus grand de fonctions équivalentes pour extraire ces caractéristiques de forme. Voici quelques lignes en python pour extraire des région de pixels connexes dans une image binarisée ``` label()``` et de ces régions extraire quelques propriétés ``` regionprops()```. Dans cet exemple, nous utilisons l'orientation et le centroid. Cette fonction offrent de nombreuses autres caractéristiques que vous pouvez extraires pour augmenter la dimension du vecteur de forme et ainsi obtenir un meilleur espace de représentation et de classification des formes en présence (https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops).

```
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
import math

# Si thresh est l'image binarisée
# extraction des régions et des propriétés des régions
label_img = label(thresh)
regions = regionprops(label_img)
print(regions)
cv2.waitKey(0)

# affichage des régions et des boites englobantes
fig, ax = plt.subplots()
ax.imshow(thresh, cmap=plt.cm.gray)

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

    ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
    ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
    ax.plot(x0, y0, '.g', markersize=15)

    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-b', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()

cv2.waitKey(0)
```

## Reconnaissance d'objets par feature matching

L'appariement de caractéristiques est un groupe d'algorithmes qui jouent un rôle important dans certaines applications de vision par ordinateur. L'idée principale est d'extraire des caractéristiques particulières d'une image d'entraînement (qui contient un objet spécifique) et d'extraire ces mêmes caractéristique d'une autre images comportant ou non cet objet afin de le retrouver et de le localiser le cas échéant. Si des similitudes existent entre la distribution spatiale de ces caractéristiques alors il est probable que l'objet se trouve dans l'image testée.

Cet ensemble de caratéristiques doit être un attribut le plus distinctif de l'objet considéré. Si vous souhaitez détecter plusieurs objets alors chacun d'entre eux doit être caractérisé par un attribut unique.

Ces caractéristiques peuvent être locales i.e. représentatives d'une distribution de couleur ou de niveau de gris autour d'un pixel particulier (un coin, un contour) c'est ce qu'on appelle un point d'intéret. Ces points d'intérêt sont ensuite décrits par un vecteur (le descripteur) dont les composantes sont calculées à partir du voisinage autour du point d'intéret. Le descripteur doit être invariant à certains changements que peut subir l'image de l'objet à retrouver : changement d'échelle, rotation, changement de luminosité. Ainsi, ces points d'inérêt et les descripteurs correspondants sont extraits de l'image modèle et de l'image test pour être finalement comparés afin de trouver des similarités.

** FAST** (Features from Accelerated Segment Test) est un premier algorithme de détection de point d'intérêt de typ coin. C'est un algrithme rapide. Voici comment faire appel à cet algrithme implanté sous OpenCV. Dans cet exemple, les points d'intérêt sont extrait avec l'étape de non-maximal suppression qui permet de supprimer les redondances trop nombreuses d'un même point d'intérêt.

```
import cv2
import numpy as np

image = cv2.imread('train.jpg')

# l'image doit être tranformée en niveau de gris
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# instanciation de la classe FAST
fast = cv2.FastFeatureDetector_create() 

# extraction des keypoints avec non Max Supression
Keypoints_1 = fast.detect(gray, None)

# création d'une copie de l'image initiale 
image_with_nonmax = np.copy(image)

# Dessiner les keypoints sur l'image initiale
cv2.drawKeypoints(image, Keypoints_1, image_with_nonmax, color=(0,35,250))

cv2.imshow('Non max supression',image_with_nonmax)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Afin de comparer, faite une extraction en gardant les redondances en remplaçant/ajoutant les lignes suivantes :
```
# non Max Supression désactivé 
fast.setNonmaxSuppression(False)

#Keypoints without non max Suppression
Keypoints_2 = fast.detect(gray, None)

```

Alors que nous avons détecté un ensemble de coins FAST, il faut désormais extraire un descripteur en chacun d'eux. **BRIEF** (Binary Robust Independent Elementary Features) est un descripteur qui converti la valeur des pixels d'un voisinage (ou patch) en un vecteur binaire (binary feature descriptor). Dans l'article scientifique de référence, les auteurs utilisent un vecteur de 128 à 512 bits.
Le script suivant utilise les lignes de codes précédentes (avec non max suppression) et y ajoute l'extraction du descripteur : 
```
import cv2
import numpy as np

image = cv2.imread('train.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
fast = cv2.FastFeatureDetector_create() 

# instanciation de la classe Brief
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

keypoints = fast.detect(gray, None)    
# Extraction des descripteurs en chaque keypoint
brief_keypoints, descriptor = brief.compute(gray, keypoints)

brief = np.copy(image)
non_brief = np.copy(image)

# Draw keypoints on top of the input image
cv2.drawKeypoints(image, brief_keypoints, brief, color=(0,35,250))
cv2.drawKeypoints(image, keypoints, non_brief, color=(0,35,250))

cv2.imshow('Fast corner detection',non_brief)
cv2.imshow('BRIEF descriptors',brief)

cv2.waitKey(0)
cv2.destroyAllWindows()
```

Certains auteurs ont proposé de combiner FAST et BRIEF mais en améliorant la robustesse de FAST aux effets d'échelle et en ajoutant une certaine invariance à la rotation et en rendant BRIEF invariant aux rotations. Il s'agit de l'algorithme **ORB** (Oriented FAST and Rotated BRIEF). Voici quelques lignes de codes pour extraire les features ORB :

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread('train.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Instanciation de la classe ORB
orb = cv2.ORB_create(nfeatures = 1000)

preview = np.copy(image)
dots = np.copy(image_1)

# Extraction des keypoints et des descripteurs
keypoints, descriptor = orb.detectAndCompute(gray, None)

# Dessine les keypoints
cv2.drawKeypoints(image, keypoints, preview, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image, keypoints, dots, flags=2)

cv2.imshow('Points',preview)
cv2.imshow('Matches',dots)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Lorsque la fonction ```ORB()``` est appliquée à l'image de l'objet à retrouver les variables ** keypoints** et **descriptor** constituent son modèle que nous allons tenter de retrouver grâce à une fonction de mise en correspondance (```bruteForce()```) grâce aux lignes suivantes. Cette fois, nous allons chargé deux images desquelles nous allons extraire les keypoints et les descripteurs que nous allons ensuite tenter de matcher.

```
import cv2
import matplotlib.pyplot as plt
import numpy as np

image_1 = cv2.imread('train.jpg')
image_2 = cv2.imread('test.jpg') 

gray_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_RGB2GRAY)

orb = cv2.ORB_create(nfeatures = 1000)

# Copie de l'image originale poru afficher les keypoints
preview_1 = np.copy(image_1)
preview_2 = np.copy(image_2)

# copy d'image_1 pour afficher les points uniquement
dots = np.copy(image_1)

train_keypoints, train_descriptor = orb.detectAndCompute(gray_1, None)
test_keypoints, test_descriptor = orb.detectAndCompute(gray_2, None)

#Draw the found Keypoints of the main image
cv2.drawKeypoints(image_1, train_keypoints, preview_1, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image_1, train_keypoints, dots, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

#############################################
########## Mise en correspondance ###########
#############################################

# Instanciation the BruteForce Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

# Lancement du match à partir des deux images
matches = bf.match(train_descriptor, test_descriptor)

# Nous trions les meilleurs matchs pour finalement ne garder que les
# meilleurs matchs (valeur les plus faibles). Ici nous gardons les 100
# premiers pour notre exemple.

matches = sorted(matches, key = lambda x : x.distance)
good_matches = matches[:100]

# Récupère les keypoints des good_matches
train_points = np.float32([train_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
test_points = np.float32([test_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# Il faut trouver la transformation homographique qui permet de passer
# de l'ensemble des keypoints de l'image de training à l'ensemble
# des keypoints de l'image de test afin de dessiner la boite englobant
# l'objet dans l'image test

# En utilisant un RANSAC
M, mask = cv2.findHomography(train_points, test_points, cv2.RANSAC,5.0)

h,w = gray_1.shape[:2]

# Création de la matrice à partir du résultat du ransac
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

# Dessine mes point matchés
dots = cv2.drawMatches(dots,train_keypoints,image_2,test_keypoints,good_matches, None,flags=2)

# Dessine la boite englobante
result = cv2.polylines(image_2, [np.int32(dst)], True, (50,0,255),3, cv2.LINE_AA)

cv2.imshow('Points',preview_1)
cv2.imshow('Matches',dots)
cv2.imshow('Detection',result)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Appliquer le même script sur les images sign_stop.png (train) et (roadsign.png). Attendion, il faudra peut être changer le nombre de keypoints matchés (100 dans l'exemple précédent) afin d'être le plus précis dans la détection.
  
Modifier ce script en appliquant la reconnaissance sur les images provenant de la caméra de votre PC et à partir d'un objet "model" (un mug avec image par exemple) que vous aurez préalablement choisi, présenté devant la caméra et modélisé par l'algorithme ORB.
L'idée est de dessiner une boite englobante autour de ce même objet lorsqu'il est détecté dans le flux de la caméra.

## Détection de QRcode

Pour terminer ce TP, je vous propose d'utiliser le module d'OpenCV (aruco library) qui permet de gérer les QRcodes (création et détection). Pour créer des images de QRcode. Il en crée 5 relatives aux chiffres allant de 1 à 5.
```
import cv2 
import numpy as np 
from cv2 import aruco

#Initialize the dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
for i in range (1, 5):
    size = 700
    img = aruco.drawMarker(aruco_dict, i, size)
    cv2.imwrite('./image_'+str(i)+".jpg",img)    
    cv2.imshow('artag',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
```
Une fois que vous avez générés certains codes, il est possible d'en détecter et localiser certains autres.

```
import cv2 
import numpy as np 
from cv2 import aruco

image = cv2.imread('exemple1.jpg')
h,w = image.shape[:2]

image = cv2.resize(image,(int(w*0.7), int(h*0.7)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Init du dictionnaire aruco et lancement de la détection
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# Détection des coins et des id
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Affichage des markers
frame_markers = aruco.drawDetectedMarkers(image.copy(), corners, ids)

cv2.imshow('markers',frame_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Pour accéder aux coins de chaque QRcode : ```c=corners[i][0]```. **c[:,0]** et **c[:,1]** sont les coordonnées des 4 coins de chaque codes.

Modifier le script précédent pour tracer un cercle de rayon 10  au centre de chaque code : ``` cv2.circle()```.

Pour finir, nous allons utiliser les QRcodes pour faire de la réalité augmentée.  Comme sur l'image exemple2.jpg, en plaçant un certains nombre de  QRcodes, il est possible de défnir une zone dans laquelle il est possible d'agir par exemple en insérant une image voire une vidéo. Nous allons réaliser un tel script à partir de tout ce qu'on a fait auparavant. Voici le script qui insère l'image de la terre dans l'espace délimitée par le centre des 4 QRcodes.

```
import os
import cv2 
import numpy as np 
from cv2 import aruco

def order_coordinates(pts, var):
    coordinates = np.zeros((4,2),dtype="int")

    if(var):
        #Parameters sort model 1 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[3] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[2] = pts[np.argmax(diff)]
    
    else:
        #Parameters sort model 2 
        s = pts.sum(axis=1)
        coordinates[0] = pts[np.argmin(s)]
        coordinates[2] = pts[np.argmax(s)] 

        diff = np.diff(pts, axis=1)
        coordinates[1] = pts[np.argmin(diff)]
        coordinates[3] = pts[np.argmax(diff)]
    
    return coordinates

image = cv2.imread('exemple1.jpg')
h, w = image.shape[:2]

image = cv2.resize(image,(int(w*0.7), int(h*0.7)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Initialise une liste vide pour garder les coordonnées du centre de
# chaque QRcode
params = []

for i in range(len(ids)):

    # Récupère les coins de chaque tag
    c = corners[i][0]

    # Dessin un cercle en chaque centre
    cv2.circle(image,(int(c[:, 0].mean()), int(c[:, 1].mean())), 3, (255,255,0), -1)
    
    # Sauvegarde les coordonnées de chaque centre dans params
    params.append((int(c[:, 0].mean()), int(c[:, 1].mean())))

# Transfome la list en array
params = np.array(params)

# Cette étape est indispensable pour ordonner les centres
# car le détecteur les trouve mais dans un ordre quelconque
if(len(params)>=4):
    #Sort model 1 
    params = order_coordinates(params,False)
    
    #Sort Model 2
    params_2 = order_coordinates(params,True)

# Nous chargeons l'image que nous souhaitons insérer entre les QRcodes
paint = cv2.imread('earth.jpg')
height, width = paint.shape[:2]

# Nous définissons les coordonnées de la région d'intérêt que nous
# souhaitons afficher i.e toute l'image
coordinates = np.array([[0,0],[width,0],[0,height],[width,height]])

# Comme vous l'avez fait auparavant pour le feature matching nous 
# calculons la transformation homographique entre l'image de la terre et
# les coordonnées de la zone sur laquelle nous voulons insérer l'image en
# question
hom, status = cv2.findHomography(coordinates, params_2)
  
# Ensuite nous la transformation selon cette homographie
warped_image = cv2.warpPerspective(paint, hom, (int(w*0.7), int(h*0.7)))

# Nous créons un mask de l'image initiale dans laquelle les pixels
# appartenant à la zone comprise entre les QRcodes sont à 255 et les autres à 0
mask = np.zeros([int(h*0.7), int(w*0.7),3], dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32([params]), (255, 255, 255), cv2.LINE_AA)

# combinaison entre le mask et l'image initiale
substraction = cv2.subtract(image,mask)

# insertion de l'image warpé dans la zone blanche. 
addition = cv2.add(warped_image,substraction)

cv2.imshow('black mask',mask)
cv2.imshow('detection',addition)
cv2.imshow('substraction',substraction)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Après avoir bien compris chaque étape de ce script, modifer le afin d'insérer les images provenant de votre flux caméra ou celles de la video chris.mp4 ou de toute autre vidéo en votre possession..

## Classification d'images par la mathode des K plus proches voisins (k-NN ou KNN)

Cet exercice permettra d'apprendre un modèle à partir des images de la bases CIFAR-10 téléchargeable ici:
http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

Décompresser les fichier dans un dossier que vous utiliserez dans le script suivant.
Ici, le dossier est ./data (par exemple)

Dans ce script, nous utilisons le modul pickle. Le module pickle implémente des protocoles binaires de sérialisation et dé-sérialisation d'objets Python.
La sérialisation est le procédé par lequel une hiérarchie d'objets Python est convertie en flux d'octets. La désérialisation est l'opération inverse,
par laquelle un flux d'octets (à partir d'un binary file ou bytes-like object) est converti en hiérarchie d'objets. Ici les fichiers binaires cifar regroupent
l'ensemble des images et leurs annotations.

```
import numpy as np
import cv2

basedir_data = "./data/"
rel_path = basedir_data + "cifar-10-batches-py/"

# Désérialiser les fichiers image afin de permettre l’accès aux données et aux labels:

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    return dict

X = unpickle(rel_path + 'data_batch_1')
img_data = X[b'data']
img_label_orig = img_label = X[b'labels']
img_label = np.array(img_label).reshape(-1, 1)
````

afin de vérifier que tout s'est bien passé utilisé :

```
print(img_data)
print('shape', img_data.shape)
```
Vous devriez trouver un tableau numpy de 10000x3072 d'uint8s (le 3072 vient du 3 x 1024). Chaque ligne du tableau stocke une image couleur 32x32 en RGB. L'image est stockée dans l'ordre des lignes principales, de sorte que les 32 premières entrées du tableau correspondent aux valeurs des canaux rouges de la première ligne de l'image.
Pour vérifier les labels :
```
print(img_label)
print('shape', img_label.shape)
```
Nous avons les étiquettes comme dane matrice 10000 x 1

Pour charger les données de test, utiliser la même procédure que précédement car la forme des données de test est identique à la forme des données d’apprentissage:
```
test_X = unpickle(rel_path + 'test_batch');
test_data = test_X[b'data']
test_label = test_X[b'labels']
test_label = np.array(test_label).reshape(-1, 1)
```
Vérifier que tout s'est bien déroulé comme précédement : deux tableaux numpy de respectivement 10000 x 3072 et 10000 x 1 élements. 
Pour extraire les a10 premières images de img_data et vérifier la taille du contenu de chaque élément, il suffit de faire ainsi :
```
sample_img_data = img_data[0:10, :]
print(sample_img_data)
print('shape', sample_img_data.shape)
print('shape', sample_img_data[1,:].shape)
````
Attention, lors de leur enregistrement  les composantes RGB des images sont arrangées sous la forme d'un vecteur à 1 dimension c'est-à-dire comme si vous aviez placer les pixels les uns derrières les autres..
Pour afficher chaque image, il faut donc remettre sous la forme d'une image 2D RGB, c'est-à-dire les "reshaper". Pour cela, nous opérons de la manière suivante en considérant que les images avaient une résolution initiale de 32x32.

```
# affichage de l'image 0
# pour une autre image remplacer 0 par un autre entier
one_img=sample_img_data[0,:]
r = one_img[:1024].reshape(32, 32)
g = one_img[1024:2048].reshape(32, 32)
b = one_img[2048:].reshape(32, 32)
rgb = np.dstack([r, g, b])
cv2.imshow('Image CIFAR',rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

Désormais, nous allons appliquer l'algorithmes des k-NN sur toutes les images de la base de training img_data et leurs labels img_label_orig

```
from sklearn.neighbors import KNeighborsClassifier 

batch = unpickle(rel_path + 'batches.meta');
meta = batch[b'label_names']

def pred_label_fn(i, original):
    return original + '::' + meta[YPred[i]].decode('utf-8')

nbrs = KNeighborsClassifier(n_neighbors=3, algorithm='brute').fit(img_data, img_label_orig)

# test sur les 10 premières images
data_point_no = 10
sample_test_data = test_data[:data_point_no, :]

YPred = nbrs.predict(sample_test_data)

for i in range(0, len(YPred)):
    show_img(sample_test_data, test_label, meta, i, label_fn=pred_label_fn)
```

Dans ce script, vous aurez besoin des fonctions suivantes pour l'affichage. Ajouter les en début de script.

one_img = img_arr[index,:]
    # Assume image size is 32 x 32. First 1024 px is r, next 1024 px is g, last 1024 px is b from the (r,g b) channel
    r = one_img[:1024].reshape(32, 32)
    g = one_img[1024:2048].reshape(32, 32)
    b = one_img[2048:]. reshape(32, 32)
    rgb = cv2.merge([b,g,r])
    img1=cv2.resize(rgb,(240,230))
    plt.imshow(img1)
    plt.show()
    print(label_fn(index, meta[label_arr[index][0]].decode('utf-8')))
```


