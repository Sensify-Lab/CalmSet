# CalmSet

**CalmSet: A Domain-Specific Test Collection for Affective Music Retrieval for Children with ASD**

CalmSet is a domain-specific test collection designed to support research in **affective music retrieval**, **music emotion recognition**, and **human-in-the-loop annotation** for therapeutic music contexts. The dataset and accompanying code formalize a modular therapeutic music library originally developed for interactive music therapy and repurpose it as a reproducible benchmarking resource.

This repository contains the **end-to-end annotation, aggregation, and benchmarking pipeline** used to construct CalmSet, corresponding directly to the algorithms and analyses described in the paper.

---

## **Repository Structure**
CalmSet/
├── CLAP Annotations (Algorithm 1)/
├── GPT Descriptions (Algorithm 2)/
├── CrowdSourced Human Annotations/
├── Borda Aggregation (Algorithm 3)/
├── Dataset Characterization (5–5.3)/
├── Benchmarking (Section 5.4)/
├── Mturk Screen Out Survey/
├── Musical Layers/
├── README.md
├── LICENSE
└── .gitignore


---

## **Pipeline Overview**

CalmSet is constructed using a **hybrid human-in-the-loop annotation pipeline**:

1. CLAP proposes candidate affective intent labels for each audio track.  
2. GPT-based models generate auxiliary semantic descriptions conditioned on candidate labels.  
3. Crowd workers provide ranked judgments of therapeutic or affective intent *without seeing model outputs*.  
4. Human judgments are aggregated using a Borda count–based procedure to produce graded relevance annotations.  
5. The resulting dataset is evaluated using standard information retrieval benchmarks.

---

## **Descriptions**


### **Musical Layers**
The CalmSet dataset was constructed using a **modular music library design** inspired by the framework described in our UCue paper. [Paper](https://dl.acm.org/doi/10.1145/3713043.3727053).

Each therapeutic component was implemented as an independent musical layer, with **15 unique tracks per layer**. These layers represent distinct therapeutic features (e.g., harmonic texture, rhythmic pulse, ambient soundscape, melodic contour), enabling controlled manipulation of musical structure.

To generate the dataset, we developed a Python-based combinatorial pipeline that systematically enumerated all possible layer combinations. Each composition consists of a unique configuration of active layers, where layers can be selectively enabled or disabled.

From the full combinatorial space, we finalized **432 unique songs**, under the constraint that **each song contains at least one of the seven therapeutic layers active** (i.e., the null combination was excluded).

This modular construction ensures:

- Controlled diversity across musical structures  
- Systematic coverage of therapeutic feature combinations  
- Balanced representation of layered configurations  
- Reproducible stimulus generation for analysis  

The resulting dataset enables fine-grained investigation of how individual and combined musical features influence emotional perception.

---
### **Mturk Screen Out Survey/**
Contains the survey questions we deployed on Mturk to recruit participants for the Annotations. This survey consisted of a basic questionnaire which tested their musical knowledge.  

- The survey was sent out to 50 MTurk workers, and a qualification criterion was set up  
- 16 workers cleared the screening and only these workers could access and complete the HITs (Human Intelligence Tasks).  
- Each participant was paid \$1.00 per HIT, with the average task duration estimated at 3--4 minutes.  


---

### **Mturk Annotation Task Design/**
A total of 432 songs were selected for validation, each requiring annotations from 3 different workers, resulting in 1296 total annotations. We deployed 432 separate HITs on MTurk (one per song), each allowing 3 unique submissions. Workers were instructed to: 

- Listen to the full therapeutic music clip (approximately 2.5 minutes).
- Select the top three emotion labels from a list of 8 predefined options (the same set used in CLAP).
- Write a two- to three-sentence description of the song’s emotional character in their own words.  
- Rate their agreement with the GPT-generated description using a 5-point Likert scale (-2 = Strongly Disagree, 2 = Strongly Agree).

The interface was designed in such a way that the worker initially had to listen to the full song before moving forward to the annotation process. Also, the worker could not completely mute the song. Once they listened to the complete song, the worker had to choose the annotations and describe the song in 2-3 sentences. Finally, the GPT-generated description was shown, and the worker had to choose their agreement based on the Likert scale.  The figure shows the work flow and the annotation interface. 

![MTurk Interface](Mturk%20Screen%20Out%20Survey/interface.png)

Each HIT is linked to an external annotation interface hosted on our server. The annotation interface auto-logged the worker ID, song ID, timestamp, and all responses. After completing the task, workers received a unique, auto-generated completion code to paste into MTurk for payment verification. The backend system enforced constraints to prevent: (i) more than 3 annotations per song and (ii) duplicate annotations by the same worker on the same song. The hits were launched in batches of 25, and the responses were monitored after a batch was completed. The workers with poor responses were blacklisted from the next batch. This helped us in further filtering out the poor-quality workers from the pre-qualified set.

---

### **CLAP Annotations (Algorithm 1)/**
Implements *Algorithm 1* from the paper.

- Runs zero-shot inference using CLAP (*Contrastive Language–Audio Pretraining*).  
- Produces candidate affective or therapeutic intent labels.  
- Computes similarity scores between audio tracks and label prompts.  

Outputs from this stage are **not shown to human annotators** and are used only as candidate signals and baselines.

---

### **GPT Descriptions (Algorithm 2)/**
Implements *Algorithm 2* from the paper.

- Generates natural-language semantic descriptions for each track.  
- Conditions generation on the top-ranked CLAP labels.  
- Produces auxiliary textual metadata used to contextualize affective intent.  

These descriptions support analysis and benchmarking but are not treated as ground-truth labels.

---

### **CrowdSourced Human Annotations/**
Contains scripts and processing logic for collecting **ranked judgments** from qualified crowd workers.

Key properties:
- Annotators rank affective or therapeutic intent labels.  
- Annotators do **not** see CLAP predictions or GPT descriptions.  
- Multiple independent judgments are collected per track.  

This stage captures the **raw human signal** prior to aggregation.

---

### **Borda Aggregation (Algorithm 3)/**
Implements *Algorithm 3* from the paper.

- Aggregates ranked human judgments using a **Borda count–based procedure**.  
- Converts rankings into graded relevance scores.  
- Applies deterministic tie-breaking to ensure reproducibility.  

The output of this stage forms the **final gold relevance annotations**.

---

### **Dataset Characterization (5.1–5.4)/**
Analysis scripts supporting Sections *5.1–5.4* of the paper.

Includes:
- Dataset statistics and distributions,  
- Label frequency and co-occurrence analysis,  
- Structural diversity of modular compositions,  
- Agreement and consistency analyses.

---

### **Benchmarking (Section 5.5)/**
Implements retrieval benchmarks described in *Section 5.5* of the paper.

Includes:
- Baseline retrieval methods (e.g., BM25, CLAP-based retrieval),  
- Evaluation using graded relevance metrics (*nDCG@k*, *MAP@k*, *Recall@k*),  
- Macro-averaged results across affective intent queries.

This folder enables **reproducible benchmarking** using CalmSet.

---

## **Dataset Availability**

The audio files and final annotation CSVs are released separately via a public dataset repository.  
This GitHub repository contains **all code necessary to reproduce the annotation pipeline, aggregation procedure, and evaluation results**.

---

## **Intended Use**

CalmSet is intended for:
- Affective and therapeutic music retrieval research,  
- Music emotion recognition and representation learning,  
- Evaluation of zero-shot and weakly supervised audio–text models,  
- Studies of human–AI agreement in semantic music annotation.

The dataset is designed for **graded relevance evaluation**, not only binary classification.

---

## **License**

This repository is released under the **MIT License**.  
Please see the `LICENSE` file for details.

---

## **Citation**

If you use this code or the CalmSet dataset in your research, please cite the accompanying paper: TBD



---

## **Contact**

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the authors directly at sensifylab@gmail.com


---

## **Dataset Link**

The dataset can be found on Kaggle at https://www.kaggle.com/datasets/sensifylab/calmset-music-for-children-with-asd/data


