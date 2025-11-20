# Stages
## -0.5 Exploration – 48 hours
Of data: which subset to choose, what is downloaded. 
Of benchmark models: which models to assign.  

Once we all have access to data, stage 0 starts. 

## 0. Data – 1 week
Goals: 
* select subset of data
* determine access solution
* processing pipeline
* loading pipeline

Subset: TBD.

    Derirable properties: big enough for training, small enough to download locally, must contain obstructions in EGO POV. 

Access solution: either download locally or store online.

Challenge: extract frames with object.
Solution: Use narrations to find relevant frames with objects of interest.

Use version 2.0 of dataset. 

## 1. Basic correspodance – 1 week
Goal: Implement existing models on basic correspondence task.

Follow implementation instructions in READMEs of existing solutions, connecting them to our data pipeline. 

Each of us takes one baseline model.

## 2. Occluded correspodance – 2 weeks
Goal: Adapt approach to work with partial or full occlusion to achieve "Seeing through walls" superpower in Ego. 

Potential solutions:
* inject occlusions in training by generating random occlusions of objects and other image albumentations.

    Note: use same frames from stage 1 so we can compare changes. 

Other solutions:
* modify models to create richer representations across exo and ego views. 
* model objects in parts so you can detect them even when they are partially occluded
* recover features in occluded regions by generative models
* occlusion aware attention on detection heads

# Resources
## Existing models available
We are not writing any new models. Instead, we use exitsting models: 
* [XSegTx](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/SegSwap)
* [XMem](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence/XMem)
* [ObjectRelator](https://github.com/lovelyqian/ObjectRelator?tab=readme-ov-file) (2nd place of Eval competition)
* [O-MaMa](https://github.com/Maria-SanVil/O-MaMa) (1st place of Eval competition)

## Ego-Exo Correspondence
* [Correspondence task description](https://docs.ego-exo4d-data.org/benchmarks/relations/correspondence/)
* [Correspondence task repository](https://github.com/EGO4D/ego-exo4d-relation/tree/main/correspondence)
* [EgoExo4d Eval AI leaderboard](https://eval.ai/web/challenges/challenge-page/2288/leaderboard/5659)







# Project contribution – focus for last 2 weeks:
* A report on comparing model performances – comparison of baseline vs fine tune for other models.
* A report on implementing a Computer Vision SOTA model and studying results – explain implementation and study results with visuals of predictions and distributions of errors.  

* A report on researching limitation of SOTA model – manually select frames with heavy occlusion in Ego and study degradation of performance of model. 
Estrazione dei frame con oggetti occlusi: scrapre le annotations e selezionare coppie di frame dove la mashcera in exo è positiva per almeno un pixel e la maschera in ego è zero per tutti i pixel. Successivamente, selezionare manualmente i frame dove c'è occlusione completa e testare i modelli fine-tuned. Studiare le maschere predicted per intuire i limiti del modello. 
* Varianza degli oggetti attraverso tutti gli scenario. 

* Mask temporal consistency + occlusion analysis in subset.
* Altri lavori che fanno cose simili da integrare nella pipeline.
* Estrarre le feature da DINOv2 e storarle. 
* Analisi del fine tuning in base a diversi training. 
* Usarlo sui nostri dati. 
* Blocchi da altri paper. 
* Altro dataset Ego-Exo. 






Analisi per progetto
1. Training & Test time dopo swap di DinoV2 con Resnet - risolve problema di lentezza in training. La storia è di fare funzionare un modello SOTA su device più semplici. 
2. Fine tuning vs pesi normali per health e cooking, confronto sia tra DinoV2
3. Le nostre foto, parte con occlusion e parte non per testare la robustness.


Trovare dei momenti settimana prox per lavorare insieme






















