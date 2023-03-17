# Introduction en traitement et analyse des images pour des applications de robotique (Partie 3)

Attention cette année ce TP pourrait se faire sur [colab](https://colab.research.google.com/).

## Segmentation des images par la méthodes des k-moyennes (kmeans)

Kmeans est un algorithme de clustering, dont l'objectif est de partitionner n points de données en k grappes. Chacun des n points de données sera assigné à un cluster avec la moyenne la plus proche. La moyenne de chaque groupe s'appelle «centroïde» ou «centre». Globalement, l'application de k-means donne k grappes distinctes des n points de données d'origine. Les points de données à l'intérieur d'un cluster particulier sont considérés comme «plus similaires» les uns aux autres que les points de données appartenant à d'autres groupes. Cet algorithme peut être appliquer sur des points d’origine géométrique, colorimétriques et autres. 

Nous allons appliquer cette méthode afin d'assurer une segmentation couleur d'une image i.e. cela revient à trouver les couleur domainantes dans l'image.

```
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
import numpy as np

#Ensuite charger une image et la convertir de BGR à RGB si nécessaire et l’afficher :
image = cv2.imread('fleur.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure()
plt.axis("off")
plt.imshow(image)
```

Afin de traiter l’image en tant que point de données avec la fonction de clustering, il faut la convertir la forme matricielle de l'image en une forme vectorielle :

```
n_clusters=5
image = image.reshape((image.shape[0] * image.shape[1], 3))
clt = KMeans(n_clusters = n_clusters )
clt.fit(image)
```

Pour afficher les couleurs les plus dominantes dans l'image, il faut définir deux fonctions : centroid_histogram() pour récupérer le nombre de clusters
différents et créer un histogramme basé sur le nombre de pixels affectés à chaque cluster ; et plot_colors() pour initialiser le graphique à barres représentant la fréquence relative de chacune des couleurs

```
def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    return bar
```

Il suffit maintenant de construire un histogramme de clusters puis créer une figure représentant le nombre de pixels étiquetés pour chaque couleur.

```
hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
```

## Utilisation des réseaux convolutifs sous opencv

Cette section du TP n'est pas là pour vous introduire ces nouveau outils avancés et très récents de traitement et d'analyse des images.
Vous sera présenté comment OpenCV exploite certains réseaux de neurones pré-entrainés par d'autres équipes sur des divers bases d'images telles que Imagenet et COCO.
OpenCV (dans sa version 3.x que j'utilise pour ces TP) permet de travailler avec de réseaux de neurones qui ont été designés dans les frameworks Caffe, Tensorflow et Troch/Pytorch.

La librairie OpenCV qui offre les fonctionnalités neuronales est ```cv2.dnn```

Pour charger les images que vous allez mettre en entrée du réseau pour inférer le résultat, vous pouvez faire appel aux fonctions ```cv2.dnn.blobFromImage()``` et ```cv2.dnn.blobFromImages()```.
Pour créer et charger des réseaux en les important de certains frameworks vous utiliserez les fonctions suivantes : 
```
cv2.dnn.readNetFromCaffe()
cv2.dnn.readNetFromTensorFlow()
cv2.dnn.readNetFromTorch()
cv2.dnn.readhTorchBlob()
```

Une fois que le modèle pré-entrainé a été chargé alors il suffit de faire appel à la méthode `.forward` de l'instanciation de la classe pour inférer le réseau sur une image. 

Exemple 1 : utilisation du réseau créer par Google (GoogLeNet) aussi connu sous le nom de Inception (https://arxiv.org/abs/1409.4842). Nous allons charger les poids du réseau
sous le format Caffe. Ce réseau permet d'associer une classe parmi les 1000 à l'image présentée à l'entrée du réseau par analyse de son contenu. Donc ce réseau ne permet pas de classer tous les objets
qu'elle contient. Il n'infère la classe que de l'objet "majoritaire".

```
import numpy as np
import argparse
import time
import cv2

# charge une image dans la variable **image**
image = cv2.imread("votreimage")

# charge les labels que le réseau a appris à reconnaitre et crée une liste
rows = open("synset_words.txt").read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
print(classes)

# Il faut savoir que  le réseau requiert une résolution specifique de ses image d'entrée
# donc l'image que vous allez soumettre au réseau doit étre resizée à la résolution 224x224 pixels.
# Par ailleurs, il faut également normaliser l'image que vous soumettez en soustrayant la valeur moyenne
# des niveaux des canaux RGB de la base d'apprentissage : ici (104, 117, 123). 
# la variable **blob** a le shape suivant : (1, 3, 224, 224)
blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

# Chargement du modèle que je vous ai fourni
# Le fichier .prototxt décrit l'architercture du réseau
# Le fichier .caffemodel est l'ensemble des poids du réseau après apprentissage
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("bvlc_googlenet.prototxt", "bvlc_googlenet.caffemodel")

# On place l'image à l'entrée du réseau et on lance l'inférence pour obtenir le résultat de la classification.
net.setInput(blob)
start = time.time()
preds = net.forward()
end = time.time()
print("[INFO] classification took {:.5} seconds".format(end - start))

# La varibale **preds** regroupe l'ensemble des prédictions.
# Elles sont classées par ordre décroissant et on ne retient que les 5 probabiltés les plus élevées. 
idxs = np.argsort(preds[0])[::-1][:5]

# Affichage des 5 meilleures prédictions
for (i, idx) in enumerate(idxs):
	# draw the top prediction on the input image
	if i == 0:
		text = "Label: {}, {:.2f}%".format(classes[idx],
			preds[0][idx] * 100)
		cv2.putText(image, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
	# display the predicted label + associated probability to the
	# console	
	print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
		classes[idx], preds[0][idx]))
# display the output image
cv2.imshow("Image", image)
cv2.waitKey(0)

```
Voici un second exemple qui permet cette fois de détecter et classer plusieurs objets dans une image ou dans une vidéo.
Le réseau utilisé est le MobileNetSSD. Le réseau peut reconnaitre 20 classes d'objets. Là encore, le réseau a été développé sous Caffe.
Le script suivant opère de la même manière que le précédent. La particularité est que la liste des classes est renseigné en dur 
dans le code.

```
import numpy as np
import argparse
import cv2 

# les labels appris par le réseau
classNames = { 0: 'background',
    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
    14: 'motorbike', 15: 'person', 16: 'pottedplant',
    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor' }

# Ouvre la vidéo ou votre caméra. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

#Load the Caffe model 
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()


    # Il faut savoir que  le réseau requiert une résolution specifique de ses image d'entrée
	# donc l'image que vous allez soumettre au réseau doit étre resizée à la résolution 300x300 pixels.
	# Par ailleurs, il faut également normaliser l'image que vous soumettez en soustrayant la valeur moyenne
	# des niveaux des canaux RGB de la base d'apprentissage : ici (127.5, 127.5, 127.5). Par ailleurs, il 
	# applique un scale factor 1/sigma = 0.007843. Ce sigma a été aussi calculé sur les composantes couleur
	# des images de la base d'apprentissage.
	# la variable **blob** a le shape suivant : (1, 3, 300, 300)
	blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
	
    # On place l'image à l'entrée du réseau et on lance l'inférence pour obtenir le résultat de la classification. 
    net.setInput(blob)
    detections = net.forward()

    cols = 300
    rows = 300

    # Récupération de l'id de la classe et de la position de l'objet détecté dans la variable **detections**.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #Confidence of prediction 
        if confidence > args.thr: # Filter prediction 
            class_id = int(detections[0, 0, i, 1]) # Class label

            # Position de l'objet
            xLeftBottom = int(detections[0, 0, i, 3] * cols) 
            yLeftBottom = int(detections[0, 0, i, 4] * rows)
            xRightTop   = int(detections[0, 0, i, 5] * cols)
            yRightTop   = int(detections[0, 0, i, 6] * rows)
            
            # Prise en compte du changement de taille de l'image
            # recalcule des bouding boxes
            heightFactor = frame.shape[0]/300.0  
            widthFactor = frame.shape[1]/300.0
            
            xLeftBottom = int(widthFactor * xLeftBottom) 
            yLeftBottom = int(heightFactor * yLeftBottom)
            xRightTop   = int(widthFactor * xRightTop)
            yRightTop   = int(heightFactor * yRightTop)
            
            # Affichage de la bouding box
            cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
                          (0, 255, 0))

            # Afichage label et de la confience associée
            if class_id in classNames:
                label = classNames[class_id] + ": " + str(confidence)
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                yLeftBottom = max(yLeftBottom, labelSize[1])
                cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),
                                     (xLeftBottom + labelSize[0], yLeftBottom + baseLine),
                                     (255, 255, 255), cv2.FILLED)
                cv2.putText(frame, label, (xLeftBottom, yLeftBottom),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                print(label)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
```


## Détection d'objets par ondelettes de Haar

La détection d'objets à l'aide de classificateurs en cascade basés sur la décomposition en ondelettes de Haar est une méthode efficace de détection d'objets proposée par Paul Viola et Michael Jones dans leur article, "Rapid Object Detection using a Boosted Cascade of Simple Features" en 2001. Il s'agit d'une approche basée sur l'apprentissage automatique où un la fonction en cascade est formée à partir d'un grand nombre d'images positives et négatives.
Cette méthode a été initialement mise en au point pour détecter des visages et a été étendu à d'autres objets tels quels les voitures.

En python, vous pouvez faire appel à cette méthode via ``` object_cascade=cv2.CascadeClassifier() ```. Cette classe est instanciée en lui passant un paramètre qui représente le "modèle" adapté à l'objet à détecter.
Vous pouvez télécharger les modèles relatifs à des humains ici : https://github.com/opencv/opencv/tree/master/data/haarcascades
Pour tester le détecteur sur des véhicules, le modèle proposé par Andrews Sobral est téléchrgeable ici : https://github.com/andrewssobral/vehicle_detection_haarcascades/blob/master/cars.xml

Pour appliquer le détecteur à une image il suffit d'appeler la méthode ```object=object_cascade.detectMultiScale(gray, scaleFactor=1.10, minNeighbors=3)``` en passant en paramètre le nom de la variable image (gray) qu'il faut préalablement transformée en niveau de gris. Il fauat également renseigner le facteur d'échelle (scaleFactor) utilisé pour réduire l'image à chaque étage et le nombre de voisins (minNeighbors) que chaque objet détecté doit avoir pour le valider comme "effectivement" l'objet recherché.

Cette méthode fournit une liste de boites englobantes (x, y, w et h) que vous afficherez sur chaque image couleur traitée afin de visualiser les résultats de la détection.

```
for x, y, w, h in object:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
```

Ecrire un script permettant de mettre en musique cette classe et cette méthode sur la vidéo cars.mp4 fournies.
Vous validerez votre script en utilisant les modèles relatifs au corps humains et en utilisant le flux d'une caméra.

### Model training

Cette méthode pourrait être très intéressante pour détecter des objets lors du "challenge". Pour cela, je vous invite à lire et utiliser ce qui est proposé sur les 2 liens suivants. Ces liens décrivent comment il est possible d'apprendre un modèle spécifique à un objet donné.

http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html

https://github.com/mrnugget/opencv-haar-classifier-training

### Model training for an other feature

Vous trouverez dans le lien suivant, l'apprentissage d'un modèle sur la base d'un autre type de caractéreristique : les Local Binary Pattern (LBP).

https://medium.com/@rithikachowta/object-detection-lbp-cascade-classifier-generation-a1d1a1c2d0b

