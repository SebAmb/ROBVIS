# Sujet Evaluation Python/OpenCV UV RobVis 2020-2021

L'objectif de ce mini-sujet est de développer le script python capable de 
détecter les **objets présents**  (voiture, vélo, piéton etc.) dans une séquence d'images prise à partir 
la caméra de surveillance d'un carrefour.

Pour cela vous vous appuierez sur les codes que vous avez testés durant les séances de TP précédentes et 
tout particulièrement ce qui vous a été montré en matière de segmentation d'objets. Il ne s'agira pas d'identifier 
la classe des objets mais simplement de les localiser au moyen d'une bounding box que vous tracerez autour de l'objet.

Pour y parvenir, je vous propose de suivre les étapes suivantes.

## Etape 1

Afin de détecter les objets présents dans une l'image *t*, un moyen simplem est de faire la différence de cette image *t*
avec une image du "fond" vide i.e. acquise sans objet présent. Dans le cadre de ce carrefour, l'image de fond (***BG.png***)
est l'image du carrefour sans aucun mobile le traversant. Pour faire une différence entre deux images vous utilserez la fonction 
```cv2.
