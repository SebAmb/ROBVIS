# Sujet Evaluation Python/OpenCV UV RobVis 2020-2021

# Sujet Evaluation Python/OpenCV UV RobVis 2020-2021

Ce mini-sujet d'une heure comporte deux volets. Dans le premier (Etape 1 à 3), il s'agit de développer le script python capable de détecter les **objets présents**  (voiture, vélo, piéton etc.) dans une séquence d'images prise à partir la caméra de surveillance d'un carrefour. Il ne s'agira pas d'identifier la classe des objets mais simplement de les localiser dans chaque image de la séquence. Dans le second volet (Etape 4 à 5), je vous demande d'écrire le script capable de retrouver deux véhicules particuliers dans une séquence.

Pour cela, vous vous appuierez très majoritairement sur les codes que vous avez testés durant les séances de TP précédentes et tout particulièrement ce qui vous a été montré en matière de segmentation et de détection d'objets.

Vous ferez tous vos tests sur la vidéo que je vous ai fournie (***carrrefour.mp4***).

Le sujet peut vous sembler dense (et il l'est sûrement) mais je pense l'avoir très bien dirigé pour que vous vous en sortiez parfaitement bien.

## Volet 1 - étape 1

Afin de détecter les objets présents dans une image *t* de la séquence, un moyen simple est de faire la différence entre cette image *t*  et une image du "fond" vide i.e. acquise lorsqu'aucun objet n'est présent. Dans le cadre de ce carrefour, l'image de fond (***bg.png***) est l'image du carrefour sans aucun mobile le traversant. Pour faire une différence entre deux images vous utilserez la fonction ```cv.absdiff(im1,im2)``` qui calcule la valeur absolue de la différence de deux images : *diff* pour faire la différence et *abs* pour en faire la valeur absolue et éviter les valeur négative. Par conséquent, il est assez facile d'imaginer qu'elle sera le résultat d'une telle différence entre l'image d'une scène vide et l'image de la même scène avec des objets.

Attention, les images de la vidéo sont de résolution élevée et en couleur. Vous veillerez à réduire la taille des images en 640x480 et à les convertir en niveau de gris.

Dans cette première étape je vous demande donc de produire le script python qui réalise les actions suivantes :

- le chargement de l'image de fond ;
- l'ouverture et la lecture de la vidéo ;
- la différence entre l'image de fond et chaque image de la vidéo ;
- l'affichage de l'image et de l'image différence dans deux fenêtres différentes.



## VolVoici les étapes.et 1 - étape 2

Dans l'étape 1, vous avez réalisé la détection des objets présents dans la scène. Dans cette deuxième étape, je vous demande :
- de procéder (par binarisation) à la création du masque de chaque objet ;
- de nettoyer (supprimer les petites régions, le bruit etc.) ce masque avec des opérateurs morpholgiques (filtre médian, dilatation, fermeture etc.). Veillez à bien remplir toutes les formes ainsi détectées  ;
- de segmenter ce masque en plusieurs régions de pixels connexes ; 
- d'afficher tous les contours en vert sur l'image (taille réduite).

## Volet 1 - étape 3

Je vous demande de modifier la partie du code de l'étape 2 afin :
- d'appliquer le masque à l'image (taille réduite) et d'afficher le résultat (vous devriez avoir en sortie une image dans laquelle seuls les objets sont présents et le reste des pixels sont noirs) ;
- dessiner et visualiser le contour des véhicules motorisés en rouge et les piétons en vert sur l'image couleur (taille réduite). Pour cela vous pourrez extraire des caractéristiques de forme sur les ensembles connexes trouvés précédemment ;
- de compter dans chaque image et en les différenciant les véhicules motorisés et les piétons/vélos. Vous afficherez ces deux compteurs en haut à gauche de l'image couleur.

*A la fin de cette étape vous enregistrerez votre script sous le nom* **volet1_nom.py **

## Volet 2 - étape 4 

Vous entrez dans le volet 2. Dans cette étape je vous demande de modifier je vous demande d'écrire un script capable de retrouver l'un ou l'autre des deux véhicules dans la séquence. Ces deux véhicules sont définisi les images suivantes : **moto.png** et **voiture.png**. Vous partirez du code fournis en TP2 et vous l'adapterez afin qu'il applique cette reconnaissance à chaque image de la séquence. Vous êtes libre d'utiliser soit une méthode fondée sur le matching d'image, soit une technique qui exploite le matching de feature. Vous veillerez à afficher une boite englobante autour des objets lorsque vous estimerez les avoir retrouvés dans les images.

*A la fin de cette étape vous enregistrerez votre script sous le nom* **volet2_1_nom.py **



## Volet 2 - étape 5

Vous aurez constaté que la redétection n'est pas toujours de bonne qualité en appliquant ces méthode de  reconnaissance à toute l'image. Dans cette étape, je vous demande de modifier le script de l'étape 4 afin que la reconnaissance soient opérée sur l'image à laquelle vous avez appliqué le mask de segmentation (cf. résultats de la question 1 du Volet 1 - etape 3). En réduisant ainsi le nombre de pixels d'intérêt, la redétection sera plus efficace.

A la fin de cette étape vous enregistrerez votre script sous le nom* **volet2_2_nom.py ** 



## A fournir en fin d'évaluation

Vous m'enverrai par email (sebastien.ambellouis@univ-eiffel.fr et sebastien.ambellouis@imt-lille-douai.fr) un fichier zip dans lequel vous placerez vos trois scripts Python. Vous avez jusque 16h30 pour me l'envoyer. Pour ceux qui utilisent Google Colab, enregistrez ces 3 mêmes fichiers mais au format notebook que vous mettrez et m'enverrez dans un fichier zip également.



