# flappy-RL

Voici un algorithme de deep q learning apprenant à jouer à une version de flappy Bird homemade.
J'ai essayé de coller le plus possible aux méchaniques de base du jeu tout en négligeant volontairement la partie graphique.

Le modèle en lui même est composé de 3 convolution et deux fully connected. 
Il prend en entrée une image de 
80x80x4 => 20x20*32 => 5x5x64 => 3x3*x4
puis un fully connected de [1600,512] et un autre de [512,2] pour l'estimation de la q value des deux actions possibles

Après 3,6m de timesteps l'algorithme atteint un niveau plus que décent, son record est de 203 à l'heure actuelle.


Voici en exclusivité un superbe 83:
https://www.youtube.com/watch?v=X7Q_cX7BxoU&feature=youtu.be
