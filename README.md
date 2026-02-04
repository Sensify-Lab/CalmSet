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
├── Dataset Characterization (5.1–5.4)/
├── Benchmarking (Section 5.5)/
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

## **Folder Descriptions**


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


