# Stages
## 0. Data – 1 week
Goals: 
* select subset of data
* determine access solution
* processing pipeline
* loading pipeline

Subset: TBD. 

    Derirable properties: big enough for training, small enough to download locally, must contain obstructions in EGO POV. 

Access solution: either download locally or store online.

## 1. Basic correspodance – 1 week
Goal: Implement existing models on basic correspondence task.

Follow implementation instructions in READMEs of existing solutions, connecting them to our data pipeline. 

Each of us takes one baseline model.

## 2. Occluded correspodance – 2 weeks
Goal: Adapt approach to work with partial or full occlusion to achieve "Seeing through walls" superpower in Ego. 

Potential solutions:
* inject occlusions in training by generating random occlusions of objects and other image albumentations.
* modify models to create richer representations across exo and ego views. 

Other solutions:
* model objects in parts so you can detect them even when they are partially occluded
* recover features in occluded regions by generative models
* occlusion aware attention on detection heads

Test model performance when reinforced with occlusion. 

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