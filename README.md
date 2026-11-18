# TP — Knowledge Distillation

TP de Master 2 VMI sur la **distillation de connaissances pour la classification d’images** avec PyTorch.

## Objectif

Je compare un **ResNet50 professeur** et des **ResNet18 étudiants** sur CIFAR-10. Je cherche à voir si la distillation améliore la précision tout en gardant un modèle étudiant plus léger. Je teste aussi des CNN personnalisés de différentes tailles.

## Méthode

- **Données :** CIFAR-10, avec 45 000 images d'entraînement, 5 000 de validation et 10 000 de test, redimensionnées en 128 × 128.
- **Stratégie 1 :** ResNet18 pré-entraîné sur ImageNet, avec et sans distillation.
- **Stratégie 2 :** ResNet18 entraîné de zéro, avec et sans distillation.
- **Distillation :** comparaison des scores, puis des scores et des *feature maps*. Les distances testées sont KL divergence, MSE et distance cosinus.
- **CNN personnalisés :** un professeur et des étudiants plus petits (largeurs divisées par 2, 3 et 4).

Les expériences du notebook ont été réalisées sur **2 époques**. Les résultats restent donc préliminaires.

## Évaluation

Je compare la **précision**, la **Cross Entropy** et les métriques d'efficacité du cours : nombre de paramètres, taille du modèle (MB), temps d'entraînement, temps moyen d'inférence par image, débit (FPS) et FLOPs estimés.

## Résultats principaux

| Modèle | Précision test |
|---|---:|
| ResNet50 professeur | 95,48 % |
| ResNet18 pré-entraîné | 92,98 % |
| ResNet18 pré-entraîné + distillation KL | **94,41 %** |
| ResNet18 entraîné de zéro | 70,02 % |
| ResNet18 entraîné de zéro + scores et *feature maps* | 71,49 % |

La distillation avec KL améliore la précision du ResNet18 pré-entraîné de **1,43 point** par rapport à son entraînement sans distillation. La première stratégie donne de meilleurs résultats que la deuxième après 2 époques.

## Notebook

📓 **[tp-knowledge-distillation.ipynb](tp-knowledge-distillation.ipynb)**

Pour reproduire le TP, ouvrir le notebook sur **Kaggle**, activer un GPU et Internet, puis exécuter les cellules dans l'ordre. Les bibliothèques nécessaires sont installées dans le notebook.

Le notebook contient les courbes d'apprentissage, les graphiques de comparaison, les matrices de confusion et les tableaux de résultats. Il enregistre les résultats dans `/kaggle/working/kd_final/` pendant l'exécution.

Les mesures de vitesse dépendent du GPU et des conditions d'exécution, et les courbes d'apprentissage sont limitées à deux points par modèle.
