# Final Project Playbook

## 1. Final Title

**Beyond LLMs: Speaker-Independent Speech Emotion Recognition with wav2vec2 and Latent Emotion Direction Analysis**

---

## 2. Executive Summary

This project will build an audio-only speech emotion recognition system on the `RAVDESS` speech dataset, using a lightweight CNN baseline and a main transformer model based on `facebook/wav2vec2-base`. We will not train a model from scratch. Instead, we will start from a pretrained self-supervised speech encoder and **fine-tune it ourselves** on a **speaker-independent** `RAVDESS` split.

The main contribution of the project is **not** the classifier alone. The distinctive part is that we will inspect the transformer's hidden representations and adapt the general logic of Anthropic's recent emotion-vector work to a speech setting:

1. identify internal emotion-sensitive directions
2. validate that those directions generalize
3. analyze the geometry of the latent space
4. intervene on the embedding and measure whether output probabilities shift

The project should therefore be presented as:

> We fine-tuned a pretrained speech transformer on a speaker-independent `RAVDESS` split for speech emotion recognition, then analyzed its hidden representations to test whether the model learns simple emotion-sensitive latent directions inspired by Anthropic's emotion-vector framework.

---

## 3. Locked Decisions

This section is the authoritative version of the project's final technical choices.

### 3.1 Dataset

- Primary dataset: `RAVDESS speech audio only`
- Use only the `audio-only speech` `.wav` files
- Do not use song files
- Do not use video files
- Do not mix in other datasets for the main result

### 3.2 Main model

- Backbone: `facebook/wav2vec2-base`
- Task adaptation: our own classification head
- Training strategy: our own fine-tuning on `RAVDESS`

### 3.3 Baseline

- A simple CNN on log-Mel spectrograms
- Purpose: reference point, not the headline contribution

### 3.4 Evaluation setting

- Main split protocol: **speaker-independent**
- No random utterance-level splitting for the official result

### 3.5 Main analysis angle

- Hidden-state extraction from the fine-tuned wav2vec2 model
- Layer-wise probing
- Emotion direction construction
- Geometry analysis with PCA and UMAP
- Steering-style intervention on pooled embeddings

### 3.6 Demo

- Local demo using one of the exact `RAVDESS` sentences
- Same sentence, different emotional tones
- Comparison of class probabilities across recordings

---

## 4. What We Will Claim

### 4.1 Safe claim

We will claim:

- we fine-tuned a pretrained self-supervised speech encoder for speech emotion recognition
- we evaluated it with a speaker-independent split
- we extracted hidden representations from the fine-tuned model
- we found evidence of emotion-sensitive latent structure
- we tested whether simple emotion directions can shift classifier behavior

### 4.2 Claim we will not make

We will not claim:

- the model has emotions
- we replicated Anthropic's paper
- we proved mechanistic causal circuits
- we discovered "true" emotion vectors in exactly the same sense as Anthropic
- our system can read human psychological truth from speech

### 4.3 Best short narrative

When describing the project in slides, presentations, and the report, use this story:

> We built a speaker-independent speech emotion recognition system with a fine-tuned wav2vec2 model, then asked a second question: does the model's hidden space contain simple emotion-sensitive directions that behave a bit like the emotion-vector idea Anthropic studied in language models?

---

## 5. Why This Project Is Distinctive

Speech emotion recognition by itself is not new. The project becomes distinctive because of the **combination** of decisions:

- we use a modern pretrained speech transformer rather than only handcrafted features
- we evaluate with a speaker-independent split instead of a weak random split
- we use `RAVDESS`, which has fixed lexical content and supports same-sentence analysis
- we do hidden-state analysis instead of stopping at classification accuracy
- we adapt the logic of Anthropic's emotion-vector work into an audio domain project
- we test interventions on latent representations, not just plots

The unique part of this project is therefore:

> not just "emotion detection," but "emotion-direction analysis inside a fine-tuned speech transformer."

---

## 6. Exact Dataset Choice

### 6.1 Dataset

- `RAVDESS`: Ryerson Audio-Visual Database of Emotional Speech and Song
- Official page: [RAVDESS dataset page](https://affectivedatascience.com/datasets)
- Paper: [Livingstone and Russo, 2018](https://doi.org/10.1371/journal.pone.0196391)
- Download source: [Zenodo](https://doi.org/10.5281/zenodo.1188976)

### 6.2 Why `RAVDESS`

`RAVDESS` is the best fit because:

- it is small enough for a class project
- labels are explicit and standardized
- audio files are clean and easy to parse
- the same sentence is spoken with different emotions
- multiple actors support speaker-independent evaluation
- emotion intensity labels give an optional arousal-style angle

### 6.3 Exact subset

We will use:

- modality: `audio-only`
- vocal channel: `speech`
- file type: `.wav`

We will exclude:

- song recordings
- audio-video files
- video-only assets

### 6.4 Exact fixed sentences

The speech subset uses two exact statements:

- Statement 01: `Kids are talking by the door`
- Statement 02: `Dogs are sitting by the door`

This matters a lot. It means we can analyze emotion while lexical content is held nearly constant.

### 6.5 Relevant dataset structure

Each file name encodes:

- modality
- vocal channel
- emotion
- intensity
- statement
- repetition
- actor

We will parse these directly from filenames and store them in a metadata table.

---

## 7. Final Label Schema

### 7.1 Initial emotion inventory in `RAVDESS`

The dataset contains:

- neutral
- calm
- happy
- sad
- angry
- fearful
- disgust
- surprised

### 7.2 Final project classes

We will use a 6-class label set:

- `neutral`
- `happy`
- `sad`
- `angry`
- `fearful`
- `disgust`

### 7.3 Mapping

- `neutral` <- merge `neutral` and `calm`
- `happy` <- `happy`
- `sad` <- `sad`
- `angry` <- `angry`
- `fearful` <- `fearful`
- `disgust` <- `disgust`
- drop `surprised`

### 7.4 Why we use this schema

- keeps the task manageable
- avoids overcomplicating the class space for a small dataset
- makes the demo easier to interpret
- gives us a stronger reference anchor for direction analysis by merging `neutral` and `calm`

### 7.5 Intensity handling

We will keep `intensity` as metadata even though it is not the main label.

We may use it later to ask:

- do strong-intensity clips lie farther from neutral than normal-intensity clips?
- does a rough arousal-like axis appear in the hidden space?

Important note:

- not every original label has both normal and strong intensity
- any intensity-based analysis will only be done where the comparison is valid

---

## 8. Official Evaluation Split

### 8.1 Main split

We will use a fixed speaker-independent split:

- Train actors: `01-16`
- Validation actors: `17-20`
- Test actors: `21-24`

### 8.2 Why this split

- no speaker leakage
- easy to explain
- easy to reproduce
- good enough for a class final

### 8.3 What we will not do

We will not use:

- random train/validation/test splitting across files
- splits that place the same speaker in train and test
- any evaluation protocol that inflates performance through leakage

### 8.4 Optional stronger extension

If time remains after the main result is complete, we may add:

- subject-wise 5-fold cross-validation as a supplemental robustness check

This is optional. The fixed split above is the official result.

---

## 9. Model Selection Strategy

### 9.1 Final backbone choice

We will use:

- `facebook/wav2vec2-base`

Official model page:

- [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)

### 9.2 Why this exact backbone

This is the right choice because:

- it is a pretrained raw speech encoder
- it is not already fine-tuned on `RAVDESS`
- it supports a clean and honest fine-tuning story
- it is smaller and simpler than larger XLSR variants
- hidden states are easy to access for downstream analysis
- it is modern and credible for an academic class project

### 9.3 What "true fine-tuning" means here

This matters for the project's academic honesty.

We are **not** using a speech-emotion model that someone already trained on `RAVDESS` and presenting it as ours.

We **are** doing this:

1. start from a generic pretrained speech encoder
2. add our own emotion-classification head
3. fine-tune it ourselves on our own speaker-independent `RAVDESS` split
4. evaluate it ourselves
5. analyze the hidden representations ourselves

That is a real fine-tuning pipeline.

### 9.4 Why we are not using a public `RAVDESS` checkpoint as the main model

Using a public `RAVDESS`-fine-tuned checkpoint would save time, but it weakens the project's story because:

- we could not truthfully say we fine-tuned the main model
- public checkpoints often use unclear or weaker split protocols
- the hidden-state analysis would be harder to defend as our own full pipeline

We may still inspect public checkpoints as sanity checks, but they will not be the main official model.

---

## 10. Code Reuse Strategy

We are not building every component from zero. We will use a smart middle ground.

### 10.1 What we will use directly

- `facebook/wav2vec2-base` weights
- Hugging Face `transformers` audio-classification patterns
- standard Python libraries for audio and ML

### 10.2 What we will borrow conceptually from existing SER implementations

We may borrow implementation ideas from public codebases such as:

- dataset parsing patterns
- fine-tuning script structure
- feature extraction conventions
- embedding extraction utilities
- evaluation formatting

### 10.3 What will still be our own work

- our exact label schema
- our exact speaker-independent split
- our fine-tuning run
- our evaluation results
- our hidden-state extraction pipeline
- our direction analysis
- our intervention experiment
- our report, visuals, and demo

### 10.4 How to describe this honestly

Use language like:

- `We fine-tuned facebook/wav2vec2-base on our speaker-independent RAVDESS split using an implementation built from Hugging Face audio-classification patterns and adapted from open-source SER workflows.`

Avoid language like:

- `We developed a speech transformer from scratch.`

---

## 11. Technology Stack

### 11.1 Core language

- Python

### 11.2 Core ML libraries

- PyTorch
- Hugging Face `transformers`
- torchaudio
- librosa
- scikit-learn
- numpy
- pandas

### 11.3 Visualization and analysis

- matplotlib
- seaborn
- plotly optional
- umap-learn

### 11.4 Demo

- Gradio preferred
- Streamlit acceptable

### 11.5 Environment guidance

Recommended:

- any machine with a CUDA-capable GPU

Acceptable alternatives:

- cloud notebook or hosted GPU environment
- smaller batch sizes with gradient accumulation

---

## 12. Repository Structure We Should Build

The repo should be organized so the project is easy to understand, rerun, and present.

### 12.1 Proposed directory structure

```text
FinalProject/
  data/
    raw/
    processed/
    metadata/
  notebooks/
    01_eda.ipynb
    02_baseline_results.ipynb
    03_wav2vec_results.ipynb
    04_emotion_vector_analysis.ipynb
  src/
    data/
      ravdess_metadata.py
      split.py
      dataset.py
      audio.py
    models/
      cnn_baseline.py
      wav2vec_classifier.py
      heads.py
    training/
      train_cnn.py
      train_wav2vec.py
      evaluate.py
      metrics.py
    analysis/
      extract_hidden_states.py
      layer_probe.py
      direction_building.py
      nuisance_control.py
      geometry.py
      steering.py
    app/
      gradio_app.py
    utils/
      seed.py
      io.py
      paths.py
      config.py
  configs/
    cnn.yaml
    wav2vec.yaml
    analysis.yaml
  artifacts/
    checkpoints/
    predictions/
    embeddings/
    plots/
    reports/
  final_project_plan.md
  README.md
```

### 12.2 Why this structure

- separates data, modeling, analysis, and app work
- makes the report easier to back up with files
- makes extraction and steering experiments reproducible
- keeps the project manageable for a team

---

## 13. Data Pipeline

### 13.1 Step 1: Download and organize raw data

Tasks:

- download `RAVDESS`
- move speech audio files into `data/raw`
- confirm no song files or video files are accidentally included
- log the dataset version and source

### 13.2 Step 2: Build metadata table

Create a CSV or parquet file with one row per sample containing:

- absolute file path
- actor id
- statement id
- repetition id
- original emotion label
- merged final label
- intensity
- split membership
- duration in seconds
- sample rate

Output file:

- `data/metadata/ravdess_metadata.csv`

### 13.3 Step 3: Quality checks

We should verify:

- no missing files
- all sample rates are readable
- label mapping is correct
- actor IDs are parsed correctly
- no speaker leakage across train/val/test

### 13.4 Step 4: Exploratory data analysis

Plots to produce:

- class counts before and after label merge
- split distribution by actor
- duration histogram
- waveform examples
- spectrogram examples by emotion
- statement distribution across labels

---

## 14. Audio Preprocessing

### 14.1 Shared audio preprocessing

All audio will be:

- loaded as waveform
- resampled to `16 kHz`
- converted to mono if needed
- normalized safely

### 14.2 Cropping and padding policy

For wav2vec2:

- keep the full utterance where possible
- use dynamic batch padding with attention masks
- do not aggressively crop speech content

For CNN baseline:

- generate fixed-size log-Mel spectrogram tensors
- pad shorter clips
- crop longer clips only if required by the baseline architecture

### 14.3 Silence trimming

Initial default:

- no aggressive silence trimming

Reason:

- timing and pauses may carry emotional information
- over-trimming may remove cues we care about

Optional controlled experiment:

- mild trimming only if initial inspection shows obvious dead air

### 14.4 Data augmentation

Start simple:

- no augmentation in the first clean run

Optional later augmentations:

- additive background noise
- slight pitch shift
- small time stretch
- mild reverb

We only add augmentation if generalization or overfitting becomes a real issue.

---

## 15. Baseline Model

The baseline is important, but it is not the main story.

### 15.1 Purpose

- provide a point of comparison
- demonstrate improvement over a simpler model
- show that the transformer result is meaningful

### 15.2 Input representation

- 64-bin log-Mel spectrogram

### 15.3 Proposed CNN architecture

- Conv block 1: conv -> batch norm -> ReLU -> max pool
- Conv block 2: conv -> batch norm -> ReLU -> max pool
- Conv block 3: conv -> batch norm -> ReLU -> max pool
- dropout
- global average pooling
- linear classifier

### 15.4 Training details

- optimizer: `Adam`
- learning rate: around `1e-3`
- loss: cross-entropy
- early stopping on validation macro F1

### 15.5 What we need from the baseline

We do **not** need the baseline to be amazing.

We do need it to:

- train cleanly
- produce nontrivial performance
- establish a fair reference point

---

## 16. Main Model: wav2vec2 Fine-Tuning

### 16.1 Backbone

- `facebook/wav2vec2-base`

### 16.2 Model structure

The main model will consist of:

1. pretrained wav2vec2 feature encoder
2. transformer stack from the pretrained model
3. pooled utterance embedding
4. dropout
5. linear classification head for 6 classes

### 16.3 Pooling strategy

Because wav2vec2 does not use a standard `[CLS]` token like BERT, we will create utterance-level embeddings by:

- taking hidden states from a chosen layer
- masking padded frames
- mean-pooling over valid time steps

This pooled representation will serve two roles:

- classification input to the head
- object of interpretability analysis

### 16.4 Fine-tuning variants

We should structure the training in stages.

#### Stage A: simple end-to-end fine-tuning

- use the pretrained backbone
- add classifier head
- fine-tune all layers

#### Stage B: stabilization fallback if needed

If training is unstable:

- freeze the feature extractor for early epochs
- fine-tune upper layers and the head first
- then unfreeze more layers gradually

### 16.5 Training objective

- cross-entropy classification loss

Optional:

- class-weighted loss if merged classes become imbalanced

### 16.6 Core hyperparameter plan

Initial defaults:

- optimizer: `AdamW`
- learning rate: `1e-5` to `3e-5`
- batch size: as large as memory allows
- gradient accumulation if needed
- dropout: around `0.1` to `0.3`
- epochs: `10-20`
- early stopping patience: around `4-5`
- best checkpoint metric: validation macro F1

### 16.7 Reproducibility rules

We should set:

- fixed random seed
- logged config files
- saved model checkpoints
- saved predictions and logits

If compute allows, run multiple seeds for the final model and report:

- mean and standard deviation

If compute is limited, run one main seed and state that clearly.

---

## 17. Official Evaluation Protocol

### 17.1 Main metrics

We will report:

- accuracy
- macro F1
- weighted F1
- per-class precision
- per-class recall
- confusion matrix

### 17.2 Most important metric

`Macro F1` should be the main comparison metric because it is more informative than raw accuracy when class balance is imperfect.

### 17.3 Evaluation outputs to save

For both baseline and wav2vec2:

- test predictions
- test probabilities
- confusion matrix image
- classification report table
- per-sample metadata joined with predictions

### 17.4 Same-sentence analysis outputs

We should also produce:

- example probability comparisons for the same sentence across emotions
- side-by-side spectrogram views
- summary plots grouped by statement ID

---

## 18. Why Speaker-Independent Evaluation Matters

This point should be emphasized in both the report and the presentation.

If we randomly split files, the model can partially exploit speaker-specific traits rather than emotion alone. That would make our performance look stronger than it really is.

Using held-out speakers means:

- the model must generalize to voices it has not seen during training
- the result is more honest
- the latent emotion analysis is more meaningful

This supports a strong final narrative:

> We did not only fine-tune on `RAVDESS`; we fine-tuned and evaluated in a speaker-independent way, then studied the learned latent space.

---

## 19. Mapping Anthropic's Emotion-Vector Logic Into Our Project

This section is the conceptual heart of the project.

### 19.1 What Anthropic did in general terms

Anthropic's paper followed a general interpretability pattern:

1. identify internal emotion-sensitive directions
2. validate that these directions activate in appropriate contexts
3. study the geometry of the representation space
4. intervene on the model using those directions
5. measure downstream effects

### 19.2 Our speech-domain translation

We will map that pattern into our project as follows:

- LLM token activations -> pooled utterance embeddings from wav2vec2
- text-based emotion contexts -> labeled emotional speech clips
- emotion vectors -> class-centroid-derived speech directions
- probe validation on text prompts -> projection validation on held-out audio
- residual-stream steering -> pooled-embedding steering before the classifier head

### 19.3 What makes this adaptation defensible

We are not claiming that speech and LLM internals are the same. We are saying that:

- both systems have internal hidden representations
- linear directions in those hidden spaces may be informative
- those directions can sometimes be studied through projection and intervention

That is the abstract idea we are borrowing.

### 19.4 Language we should use

Use:

- `Anthropic-inspired`
- `emotion-direction analysis`
- `latent-space probing`
- `embedding-space intervention`
- `representation-level analysis`

Avoid:

- `we replicated Anthropic`
- `we found the same kind of causal circuits`

---

## 20. Full Emotion-Direction Analysis Plan

This is the most important technical section in the whole playbook.

### 20.1 Inputs to the analysis

The analysis will use:

- the fine-tuned wav2vec2 model
- hidden states from all transformer layers
- train/validation/test metadata
- classification logits and probabilities

### 20.2 Step 1: Extract hidden states

For every utterance in train, validation, and test:

- run forward pass through the fine-tuned model
- save hidden states from every transformer layer
- compute attention-mask-aware mean pooling per layer
- save one fixed-size embedding per layer per utterance

Saved fields for each embedding:

- sample id
- actor id
- statement id
- repetition id
- original emotion
- final merged label
- intensity
- split
- layer index
- pooled embedding vector
- predicted label
- predicted probabilities

### 20.3 Step 2: Determine which layers matter most

We will not guess which layer to analyze. We will test it.

For each layer:

- train a simple linear classifier or linear probe on the training embeddings
- evaluate on validation embeddings
- compare macro F1 across layers

The output should be:

- a plot of layer index vs validation macro F1
- identification of one or two best layers

These best layers become the main analysis layers.

### 20.4 Step 3: Build class prototypes

Using only the training split:

- compute the mean pooled embedding for each final emotion class

This gives:

- `c_neutral`
- `c_happy`
- `c_sad`
- `c_angry`
- `c_fearful`
- `c_disgust`

Important rule:

- no test data may be used to build the direction vectors

### 20.5 Step 4: Build emotion directions

Primary directions:

- `v_angry = c_angry - c_neutral`
- `v_sad = c_sad - c_neutral`
- `v_happy = c_happy - c_neutral`
- `v_fearful = c_fearful - c_neutral`
- `v_disgust = c_disgust - c_neutral`

Why neutral is the anchor:

- it gives a stable reference state
- it makes the interpretation intuitive
- it aligns with the story of "emotion intensity away from neutral"

### 20.6 Step 5: Remove nuisance directions

This is crucial because `RAVDESS` is small and confounds can easily dominate.

Potential nuisance factors:

- speaker identity
- statement identity
- dataset-specific recording traits

We will try the following control methods.

#### Method A: speaker mean-centering

For exploratory analysis:

- subtract each speaker's mean embedding from embeddings belonging to that speaker

This asks:

- does emotion structure remain after removing speaker offset?

#### Method B: neutral-subspace PCA removal

Using neutral or low-emotion embeddings:

- compute top principal components
- project those components out of the candidate emotion directions

This follows the same general logic as removing broad non-emotional variance.

#### Method C: statement control

- compare projections within statement 01 separately
- compare projections within statement 02 separately

This tests whether a direction survives when lexical content is fixed.

### 20.7 Step 6: Validate the directions on held-out data

For each held-out test embedding:

- compute projection onto each emotion direction

We then test whether:

- angry clips project more strongly onto `v_angry` than neutral clips
- sad clips project more strongly onto `v_sad` than neutral clips
- these relationships persist within fixed statements
- these relationships persist after nuisance controls

### 20.8 Step 7: Compare projections to model probabilities

For each test utterance:

- compute direction projection scores
- compare them to softmax probabilities from the classifier head

This asks:

- if projection on `v_angry` increases, does `P(angry)` also tend to increase?

We can measure:

- Pearson correlation
- Spearman correlation
- class-separation histograms

### 20.9 Step 8: Geometry analysis

We will analyze geometry in three complementary ways.

#### A. Pairwise cosine similarity between directions

Questions:

- are negative emotions more similar to one another than to happy?
- do some emotions appear close in the hidden space?

#### B. PCA

Questions:

- does the first principal component look valence-like?
- does another component look arousal-like?

#### C. UMAP

Questions:

- do embeddings form partially separable class regions?
- do emotions with similar qualities cluster near each other?

Important caution:

- UMAP is for visualization, not proof
- statistical comparisons should still rely on actual projections and metrics

### 20.10 Step 9: Intensity analysis

Because `RAVDESS` contains normal and strong intensity for several emotions, we can test:

- whether strong examples move farther from neutral than normal examples
- whether strong clips produce larger direction projections

This gives a potential arousal-style result, but it is optional and only valid where the label structure supports it.

### 20.11 Step 10: Steering-style intervention

This is our most Anthropic-like step.

For a held-out pooled embedding `z`:

- create `z' = z + alpha * v_emotion`
- feed `z'` into the classifier head
- compare the new probability distribution to the original one

Primary questions:

- does adding `v_angry` increase `P(angry)`?
- does adding `v_happy` increase `P(happy)`?
- are the changes smooth as `alpha` changes?

Steering strengths:

- test a small range of positive and negative alpha values

Important limitation:

- this is a classifier-head intervention on pooled embeddings
- it is not full residual-stream steering across a generative model

That is okay. We should say so clearly.

### 20.12 Step 11: What would count as success

Strong result:

- one or two layers clearly encode emotion better than the rest
- held-out projections align with the correct emotions
- structure remains after nuisance control
- direction projections correlate with class probabilities
- steering shifts probabilities in the expected direction

Moderate but still valid result:

- the classifier works well
- some latent structure is visible
- projection results are partially noisy
- steering effects are small but non-random

Weak but still presentable result:

- classifier works
- latent-space effects are limited
- nuisance controls reveal that the dataset is confounded

Even this can still be turned into a good scientific discussion because it shows honest limits.

---

## 21. Statistical and Diagnostic Checks

To keep the analysis honest, we should include simple diagnostics.

### 21.1 Leakage checks

- verify no test actors appear in train or validation
- verify directions are built from train only

### 21.2 Confound checks

- predict speaker from embeddings
- compare how separable speaker identity is vs emotion
- check whether statement is easier to classify than emotion

If speaker or statement dominates the space, note it explicitly.

### 21.3 Sanity baselines

We may add simple baselines such as:

- nearest centroid by raw class mean
- linear probe without fine-tuning

These help show whether fine-tuning and direction building actually add value.

---

## 22. Output Artifacts We Must Save

This section is important because the report and slides will depend on these outputs.

### 22.1 Model artifacts

- best CNN checkpoint
- best wav2vec2 checkpoint
- training logs
- config files

### 22.2 Prediction artifacts

- test predictions CSV
- test probabilities CSV
- per-sample metadata + predictions table

### 22.3 Embedding artifacts

- pooled embeddings for each layer
- class centroid vectors
- nuisance-controlled direction vectors

### 22.4 Plot artifacts

- class distribution plots
- confusion matrices
- per-layer probe performance plot
- PCA plots
- UMAP plots
- direction projection histograms
- steering effect plots

### 22.5 Demo artifacts

- prerecorded backup samples
- screenshots or short recordings of the app

---

## 23. Demo Plan

### 23.1 Main demo goal

The audience should understand one thing immediately:

> the same sentence can be spoken with different emotional tones, and the model responds differently.

### 23.2 Sentence choice

Use one of the exact `RAVDESS` statements:

- `Kids are talking by the door`
- `Dogs are sitting by the door`

This makes the live demo much safer than using arbitrary unseen text.

### 23.3 Demo workflow

1. Open local app
2. Record the sentence in a neutral tone
3. Show predicted probabilities
4. Record the same sentence in a different tone
5. Show changed probabilities
6. Optionally show direction scores such as `angry - neutral`

### 23.4 UI elements

- record button
- waveform or playback widget
- top predicted class
- bar chart for all classes
- optional spectrogram image
- optional latent direction scores

### 23.5 Demo caution

The live demo is a presentation tool, not the official evaluation benchmark.

We should say:

- the official numbers come from held-out speakers in the test split
- the live demo is an informal real-time illustration

### 23.6 Backup plan

- have prerecorded samples ready
- test microphone beforehand
- use exaggerated emotions
- present the backup if the room is noisy

---

## 24. Report Structure

The final report should follow this structure.

### 24.1 Introduction

- motivation
- why transformers beyond LLMs
- why speech emotion recognition
- why hidden-state analysis matters

### 24.2 Related work

- speech emotion recognition
- wav2vec2 and pretrained speech models
- interpretability and latent directions
- Anthropic paper as inspiration

### 24.3 Dataset and preprocessing

- `RAVDESS`
- label mapping
- split protocol
- preprocessing choices

### 24.4 Methods

- CNN baseline
- wav2vec2 fine-tuning
- evaluation protocol
- layer probing
- direction construction
- nuisance control
- steering-style intervention

### 24.5 Results

- baseline vs wav2vec2
- confusion matrices
- same-sentence examples
- layer-wise probe plot
- PCA/UMAP
- direction projections
- steering results

### 24.6 Discussion

- what worked
- what failed
- dataset limitations
- confounds and controls
- how close our adaptation is to Anthropic's logic

### 24.7 Limitations

- small acted dataset
- not natural conversational speech
- classifier-head steering only
- not mechanistic interpretability

### 24.8 Conclusion

- summarize classifier result
- summarize latent direction result
- summarize honest takeaways

---

## 25. Presentation Structure

### 25.1 Slide story

Use this 10-step flow:

1. transformers are useful beyond LLMs
2. speech emotion recognition is a good audio-domain testbed
3. `RAVDESS` gives matched sentences and clean labels
4. we fine-tuned `facebook/wav2vec2-base` on speaker-independent splits
5. we compared it to a simple CNN baseline
6. the wav2vec2 model learned useful emotion predictions
7. we extracted hidden states from all layers
8. we found simple emotion-sensitive directions in the latent space
9. steering those directions shifts the classifier output
10. this is not proof of "model emotions," but it is evidence of useful latent emotion structure

### 25.2 Best single-sentence pitch for presentation

> Our project shows not only that a pretrained speech transformer can classify emotion from audio, but also that its hidden space contains interpretable emotion-sensitive directions that we can analyze and nudge.

---

## 26. Work Breakdown for the Team

### 26.1 Data and preprocessing owner

Responsibilities:

- dataset download
- metadata parser
- split generator
- EDA plots

### 26.2 Baseline owner

Responsibilities:

- CNN implementation
- baseline training
- baseline evaluation

### 26.3 Main model owner

Responsibilities:

- wav2vec2 classifier
- fine-tuning pipeline
- checkpointing
- final prediction export

### 26.4 Analysis owner

Responsibilities:

- hidden-state extraction
- layer probing
- direction building
- nuisance control
- geometry plots
- steering analysis

### 26.5 Demo and presentation owner

Responsibilities:

- Gradio app
- sample management
- slide visuals
- final demo rehearsal

### 26.6 If the team is small

Recommended compressed ownership:

- Person 1: data + baseline
- Person 2: wav2vec2 fine-tuning
- Person 3: hidden-state analysis + demo

---

## 27. Week-by-Week Implementation Plan

### Week 1

- finalize this playbook
- set up repo structure
- download `RAVDESS`
- build metadata parser

### Week 2

- finalize split file
- run EDA
- create baseline data loaders

### Week 3

- implement CNN baseline
- train first baseline model
- generate initial metrics

### Week 4

- implement wav2vec2 classifier
- run first fine-tuning experiment
- save first checkpoint

### Week 5

- tune wav2vec2 hyperparameters
- finalize main classifier result
- export probabilities and errors

### Week 6

- extract all-layer embeddings
- train layer-wise probes
- choose main analysis layers

### Week 7

- build class centroids
- create emotion directions
- run projection analysis
- run nuisance controls

### Week 8

- run PCA and UMAP
- run intensity analysis
- create geometry plots

### Week 9

- implement steering-style intervention
- generate steering plots
- identify best examples

### Week 10

- build demo app
- integrate best checkpoint
- connect app to probability display

### Week 11

- prepare report figures
- write methods and results sections
- rehearse live demo

### Week 12

- finalize report
- finalize slides
- dry-run presentation

---

## 28. Risk Register

### Risk 1: Small dataset causes unstable training

Mitigation:

- use pretrained wav2vec2
- keep label space to 6 classes
- use early stopping
- try freezing lower layers if needed

### Risk 2: Speaker leakage accidentally enters the split

Mitigation:

- generate split directly from actor IDs
- add validation script that asserts actor disjointness

### Risk 3: Public code causes reproducibility mess

Mitigation:

- keep a clean local training pipeline
- log configs
- save exact checkpoints and outputs

### Risk 4: Emotion directions mostly capture speaker identity

Mitigation:

- use speaker-independent split
- run speaker-centering diagnostics
- report the limitation honestly

### Risk 5: Statement identity dominates the latent space

Mitigation:

- run within-statement analyses
- build same-statement projection comparisons

### Risk 6: Steering effect is weak

Mitigation:

- keep classification as the primary technical result
- frame steering as exploratory
- emphasize layer-wise probes and nuisance-controlled projections even if steering is modest

### Risk 7: Live demo fails

Mitigation:

- use exact `RAVDESS` sentence
- keep backup prerecorded clips
- rehearse in the presentation room if possible

---

## 29. Success Criteria

The project is successful if most of the following are true.

### 29.1 Minimum success

- metadata and split pipeline are correct
- CNN baseline trains
- wav2vec2 fine-tunes successfully
- speaker-independent test metrics are reported
- demo runs locally

### 29.2 Strong success

- wav2vec2 clearly outperforms the baseline
- same-sentence examples are convincing
- one or more layers show strong linear separability
- emotion directions behave meaningfully on held-out data
- steering shifts probabilities in expected directions

### 29.3 Excellent success

- nuisance-controlled analysis still reveals emotion structure
- geometry plots show interpretable organization
- intensity analysis reveals a plausible arousal-like trend
- final presentation clearly connects seminar theme, classifier, and latent-space analysis

---

## 30. Final Deliverables

By the end of the project we should have:

1. `RAVDESS` metadata file with official split
2. EDA notebook
3. CNN baseline results
4. Fine-tuned wav2vec2 checkpoint
5. Speaker-independent test metrics
6. Hidden-state embedding cache
7. Layer-wise probe results
8. Emotion direction vectors
9. PCA and UMAP figures
10. Projection analysis results
11. Steering analysis results
12. Local demo app
13. Final report
14. Final slide deck

---

## 31. References We Will Build From

- [RAVDESS dataset page](https://affectivedatascience.com/datasets)
- [RAVDESS paper](https://doi.org/10.1371/journal.pone.0196391)
- [facebook/wav2vec2-base model card](https://huggingface.co/facebook/wav2vec2-base)
- [Hugging Face audio classification task guide](https://huggingface.co/docs/transformers/tasks/audio_classification)
- [Anthropic: Emotion concepts and their function in a large language model](https://www.anthropic.com/research/emotion-concepts-function)

Optional implementation references:

- [MMEmotionRecognition](https://github.com/cristinalunaj/MMEmotionRecognition)
- [ser-with-w2v2](https://github.com/habla-liaa/ser-with-w2v2)

These implementation references are for engineering guidance only. They are not our main scientific claim.

---

## 32. Final One-Paragraph Project Description

This project builds a speaker-independent speech emotion recognition system on the `RAVDESS` speech dataset by fine-tuning `facebook/wav2vec2-base`, a pretrained self-supervised raw speech encoder, and comparing it against a simple CNN spectrogram baseline. The main goal is not only to classify emotions such as neutral, happy, sad, angry, fearful, and disgust, but also to analyze the transformer's hidden representations and test whether they contain simple emotion-sensitive latent directions. Inspired by Anthropic's recent emotion-vector work in language models, we will extract layer-wise utterance embeddings, build class-based emotion directions, control for speaker and sentence confounds, study the geometry of the learned space, and run steering-style interventions on pooled embeddings to see whether output probabilities shift in predictable ways. This gives us a strong class project narrative that is technically meaningful, current, and more interesting than plain emotion classification.

---

## 33. Final Locked Narrative

If we need one final sentence that captures the whole project, use this:

> We fine-tuned `facebook/wav2vec2-base` on a speaker-independent `RAVDESS` split for speech emotion recognition, then studied whether the model's hidden space contains interpretable emotion-sensitive directions inspired by Anthropic's emotion-vector framework.

That is the final project plan.
