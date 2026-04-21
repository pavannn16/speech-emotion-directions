# Beyond LLMs: Transformer-Based Speech Emotion Recognition with Latent Emotion Representation Analysis

## 1. Project Overview

This project proposes a **transformer-based Speech Emotion Recognition (SER) system** that classifies human emotions directly from speech audio and supports a live demonstration using local inference. The core idea is to build a model that can distinguish emotions such as **anger, neutral, sadness, happiness, fear, and disgust** from the way a sentence is spoken, rather than from the lexical meaning of the words.

The project will be designed as a continuation of the seminar topic **"Transformers Beyond LLMs"**. Instead of using transformers for text generation, this project applies them to **audio understanding**, specifically to the task of recognizing emotional content in speech.

A second layer of the project adds a more research-oriented contribution: we will analyze whether a transformer-based audio model learns **structured internal emotion representations** in its hidden layers. This interpretability-inspired angle is motivated by recent work from Anthropic on **emotion vectors** in large language models. The goal is **not** to reproduce Anthropic's work directly, but to explore whether a similar idea can be investigated in a speech model: do hidden representations form identifiable directions or clusters corresponding to emotional states such as angry vs neutral?

The result is a project with three connected components:

1. **Main prediction system**: classify emotion from speech audio.
2. **Transformer comparison story**: show how transformer-based audio models compare to simpler baselines.
3. **Interpretability sub-study**: inspect latent emotion structure inside the model.

---

## 2. Problem Statement

Humans can often detect emotion from speech even when the words remain the same. For example, the sentence:

> "My name is Pavan, today seems like a fine weather."

can sound calm, angry, sad, or sarcastic depending on how it is spoken. The emotional signal is carried by **prosodic and acoustic features**, such as:

- pitch and pitch variation
- speaking rate
- intensity or loudness
- pauses and timing
- spectral energy distribution
- harshness or smoothness of voice quality

Traditional approaches to speech emotion recognition often rely on handcrafted acoustic features plus shallow classifiers. Modern deep learning methods instead learn these representations directly from audio. Transformers are particularly relevant because they can model **long-range temporal dependencies** and **global context**, which are important for how emotions unfold over an utterance.

The project asks the following broader question:

**Can a transformer-based speech model recognize emotions from audio robustly enough to support a live same-sentence demo, and do its internal representations exhibit interpretable emotion structure?**

---

## 3. Why This Project Fits CS 5661

This project aligns strongly with the course because it is:

- centered on **data science, machine learning, and deep learning**
- directly connected to **advanced neural architectures**, especially transformers
- built around **original scholarly work**, not just API usage
- suitable for a **group project** with modeling, experimentation, evaluation, and reporting
- presentation-friendly because it includes a clear **live interactive demo**

It also fits the course requirement that the project be **challenging enough**, related to data science, and supported by original code, report, and presentation deliverables.

---

## 4. Bridge from the Seminar: "Transformers Beyond LLMs"

This project can be positioned as a direct continuation of the seminar presentation.

### Seminar-to-project narrative

The seminar theme was that transformers are not limited to large language models. They also appear in:

- computer vision
- audio and speech processing
- multimodal learning
- time-series modeling
- recommendation systems
- robotics and agents

This project takes that exact idea into practice.

### How the bridge works

In the seminar, the focus was on the broader evolution of transformers beyond text.
In the project, the focus becomes:

- using transformers for **speech understanding** rather than text generation
- showing that attention-based architectures can model **acoustic emotion cues**
- analyzing whether transformer hidden states encode **emotion-related structure**

This lets the team say:

> Our seminar explained why transformers matter beyond LLMs. Our project demonstrates that idea through a real audio-domain application, using transformer models for speech emotion recognition and hidden representation analysis.

That creates a very clean academic narrative between seminar and project.

---

## 5. Main Research Questions

The project can be framed around the following research questions.

### RQ1. Core task
Can a transformer-based model classify speech emotion accurately from raw audio or spectrogram-based input?

### RQ2. Same-text emotion variation
Can the model distinguish the **same sentence spoken in different emotional tones**, such as neutral vs angry?

### RQ3. Speaker generalization
How well does the model generalize to **unseen speakers**, especially live recordings from a speaker not present in the training data?

### RQ4. Transformer value
Does a transformer-based model outperform or behave differently from a simpler baseline such as a CNN-only classifier?

### RQ5. Latent emotion representation
Do hidden states in the model form separable emotion clusters or directions corresponding to emotion categories?

### RQ6. Interpretability-inspired intervention
If we define simple latent directions such as **angry minus neutral**, does moving representations along those directions affect the classifier's predicted emotion probabilities?

---

## 6. Project Novelty

The project should not claim novelty simply because it performs emotion classification. Speech emotion recognition is an established task. The novelty comes from the **combination of choices and the research framing**.

### Novel elements in this project

1. **Transformer-focused SER framing** linked directly to a seminar on transformers beyond LLMs.
2. **Controlled same-sentence live demo**, where the spoken words remain constant and only the emotional tone changes.
3. **Speaker-generalization analysis**, testing whether the model works on new voices.
4. **Cross-dataset evaluation**, if time permits.
5. **Latent emotion representation analysis**, inspired by current interpretability work on emotion vectors.
6. **Steering-style exploratory experiment**, testing whether simple emotion directions in embedding space alter model predictions.

### What we should and should not claim

#### Reasonable claim
We investigate whether transformer-based speech models learn structured emotion-sensitive internal representations.

#### Overclaim to avoid
We discovered that the model literally has emotions.

The proper framing is about **representations**, **embeddings**, **latent directions**, and **classification behavior**, not human-like emotional experience.

---

## 7. Proposed System Architecture

The project can be implemented in a modular way.

### End-to-end pipeline

1. **Audio input**
   - dataset samples for training and evaluation
   - live microphone input for demo

2. **Preprocessing**
   - resampling to 16 kHz
   - silence trimming
   - normalization
   - optional augmentation (noise, pitch shift, time stretch)

3. **Representation stage**
   Choose one of the following:
   - log-Mel spectrograms
   - pretrained raw-audio transformer features
   - hybrid spectrogram + transformer pipeline

4. **Emotion classifier**
   - baseline CNN
   - CNN-Transformer hybrid
   - fine-tuned wav2vec 2.0 style model
   - optionally AST-like spectrogram transformer variant

5. **Prediction output**
   - emotion label
   - probability distribution across classes
   - confidence score

6. **Interpretability module**
   - hidden-state extraction
   - layer probing
   - centroid analysis
   - dimensionality reduction visualization
   - latent direction experiments

7. **Demo UI**
   - local web interface with microphone recording
   - prediction bars for emotion probabilities
   - side-by-side comparison of neutral vs angry demo recordings

---

## 8. Model Design Options

There are several viable model designs. The project should pick one main model and one simpler baseline.

### Option A. CNN baseline on spectrograms

#### Idea
Convert audio into log-Mel spectrograms and train a CNN classifier.

#### Advantages
- simple to implement
- strong baseline
- easier training
- useful for comparison

#### Limitation
- less aligned with the transformer theme
- weaker long-range sequence modeling

---

### Option B. CNN-Transformer hybrid

#### Idea
Use a CNN front-end to extract local patterns from spectrogram patches, then use a transformer encoder to model long-range temporal relationships.

#### Advantages
- very good bridge between classic deep learning and transformer models
- easier to explain in class
- clear architecture story
- likely feasible on local hardware

#### Limitation
- requires more design and tuning than a simple CNN

---

### Option C. Fine-tuned wav2vec 2.0 based classifier

#### Idea
Use a pretrained speech model and fine-tune it for emotion recognition.

#### Advantages
- strong research relevance
- powerful representation learning
- good option for smaller emotion datasets
- naturally transformer-based

#### Limitation
- less "from-scratch" than a custom architecture
- requires careful fine-tuning and resource management

---

### Option D. Audio Spectrogram Transformer style model

#### Idea
Treat spectrogram patches like image patches and feed them to a transformer encoder.

#### Advantages
- clean seminar connection to transformer architectures
- elegant attention-first design

#### Limitation
- more sensitive to data quantity and training stability
- may be harder to train robustly within class constraints

---

## 9. Recommended Modeling Strategy

The most practical and academically strong setup is:

### Recommended setup

- **Baseline**: CNN on log-Mel spectrograms
- **Main model**: transformer-based SER model
  - either CNN-Transformer hybrid
  - or fine-tuned wav2vec 2.0 classifier
- **Interpretability sub-study**: hidden-state and layer analysis on the transformer-based model

This gives the project:

- one simple baseline
- one advanced transformer model
- one research-oriented analysis layer

That is enough depth for the course while still remaining feasible.

---

## 10. Dataset Plan

### Primary candidate datasets

#### 10.1 RAVDESS
Useful because it contains **lexically matched statements** with multiple emotions, which fits the same-sentence demo story very well.

Best use:
- initial experiments
- same-text emotional variation analysis
- easier controlled benchmarking

#### 10.2 CREMA-D
Useful because it has more speakers and more diversity, making it better for speaker generalization.

Best use:
- broader training set
- held-out speaker testing
- stronger evaluation than RAVDESS alone

#### 10.3 IEMOCAP
Useful as a more research-heavy benchmark with multimodal and interaction-rich emotional speech.

Best use:
- advanced evaluation
- optional cross-corpus testing
- optional future extension

### Recommended practical plan

#### Minimum viable dataset plan
- train on RAVDESS and/or CREMA-D
- test on held-out speakers
- use self-recorded audio for live demo

#### Stronger research plan
- train on CREMA-D + RAVDESS
- test on held-out speakers
- optionally evaluate cross-corpus on IEMOCAP

### Label harmonization
One practical issue is that datasets do not always use exactly the same emotion labels. The team will likely need to harmonize labels to a smaller shared set, such as:

- angry
- neutral/calm
- happy
- sad
- fear
- disgust

It may also be useful to merge or remove rare/inconsistent classes depending on dataset overlap.

---

## 11. Preprocessing and Feature Engineering

Speech data must be cleaned and standardized before model training.

### Preprocessing steps

1. Resample all audio to a fixed rate, likely 16 kHz.
2. Normalize amplitude.
3. Trim leading/trailing silence.
4. Pad or crop clips to fixed duration if needed.
5. Convert labels into a common class schema.
6. Split data carefully by **speaker**, not only by utterance, to avoid leakage.

### Data augmentation ideas

- additive background noise
- random volume shift
- pitch shift
- time stretch
- reverberation simulation

These can improve robustness and help the model behave better on live recordings.

### Possible representations

#### Spectrogram-based
- log-Mel spectrogram
- MFCCs as baseline feature only if needed

#### Raw-waveform / pretrained representation-based
- wav2vec 2.0 embeddings
- hidden representations from pretrained speech encoders

---

## 12. Training Plan

### Data splitting
The correct split strategy is **speaker-independent** splitting.
That means a speaker used in training should not appear in validation or test.

This is important because random utterance-level splitting can inflate results.

### Loss function
Standard cross-entropy loss is the default choice.
Possible extensions:
- class-weighted loss for imbalance
- focal loss if imbalance is significant

### Optimization
- Adam or AdamW
- early stopping
- learning-rate scheduling
- checkpoint saving

### Metrics
- accuracy
- macro F1-score
- weighted F1-score
- confusion matrix
- per-class precision and recall

Macro F1 is especially important if classes are imbalanced.

---

## 13. Core Experimental Design

The project should be presented as a sequence of experiments rather than only a final demo.

### Experiment 1. Baseline model
Train a CNN baseline on spectrograms.

Goal:
- establish a reference point
- understand easier implementation
- build first benchmark

### Experiment 2. Transformer model
Train or fine-tune the transformer-based model.

Goal:
- compare with baseline
- connect to seminar theme

### Experiment 3. Same-sentence controlled test
Use multiple recordings of the same sentence spoken in different emotional tones.

Goal:
- isolate emotion from word content
- create a compelling demo and evaluation case

### Experiment 4. Held-out speaker evaluation
Test on speakers not seen during training.

Goal:
- study real generalization
- avoid overclaiming performance

### Experiment 5. Noise robustness
Test performance under mild environmental noise.

Goal:
- simulate realistic demo conditions
- show practical limits

### Experiment 6. Optional cross-dataset transfer
Train on one dataset and test on another.

Goal:
- make the project more research-heavy
- study domain mismatch and robustness

---

## 14. Interpretability-Inspired Sub-Study

This is the part influenced by Anthropic's recent work on emotion vectors.

### Motivation
Anthropic reported that large language models contain internal activation patterns corresponding to many emotion concepts. Their work suggests that emotional concepts can appear as **structured internal representations** rather than only as output words.

Our project uses that idea as inspiration, but in a different modality: **speech audio**.

### Our research question
Do hidden states in a transformer-based speech emotion recognition model encode distinguishable latent emotion structure?

### Practical sub-study components

#### 14.1 Embedding extraction
For each utterance, extract hidden representations from selected transformer layers.

#### 14.2 Visualization
Use one or more of the following:
- PCA
- t-SNE
- UMAP

Goal:
- visualize whether angry, neutral, happy, sad, etc. cluster separately

#### 14.3 Layer-wise probing
Train a small linear or shallow classifier on representations from each layer.

Goal:
- identify which layer carries the most emotion information

#### 14.4 Class centroid analysis
Compute average embeddings per emotion class.

Goal:
- see whether class centroids are well separated
- compute distances such as angry-to-neutral, happy-to-sad, etc.

#### 14.5 Latent emotion direction experiment
Define simple directions such as:

- angry vector = centroid(angry) - centroid(neutral)
- sad vector = centroid(sad) - centroid(neutral)

Then test a small intervention:
- move a hidden representation slightly along one of those directions
- observe whether classifier probabilities shift

### Important interpretation rule
This experiment should be presented as:

- exploratory latent-space analysis
- representation-level intervention
- interpretability-inspired probing

It should **not** be presented as proof that the model actually feels emotions.

---

## 15. Live Demo Design

The live demo is one of the strongest parts of the project.

### Demo concept
One person says the same sentence twice:

1. neutral/normal tone
2. angry tone

The model predicts different emotion probabilities for each recording.

### Why this demo is good
- it is easy for the audience to understand
- it isolates tone rather than text content
- it is interactive and memorable
- it demonstrates real local inference

### Demo pipeline
1. Open local web interface.
2. Record first version of the sentence.
3. Show predicted emotion probabilities.
4. Record second version.
5. Show changed probability distribution.
6. Optionally compare waveform or spectrogram visualizations.

### Best UI behavior
The demo should show:
- top predicted emotion
- confidence or probability bars
- optional spectrogram image
- comparison between two recordings

### Practical advice
- use exaggerated emotions for demo reliability
- keep microphone distance similar
- test in a quiet room
- have backup pre-recorded samples in case live audio fails

---

## 16. Feasibility Assessment

### Overall verdict
The project is feasible **if the scope is controlled properly**.

### Feasible version
- one baseline model
- one main transformer-based model
- one or two datasets
- one live demo interface
- one interpretability sub-study using embedding analysis

### Less feasible version
- too many datasets
- too many architectures
- full mechanistic interpretability claims
- training massive models from scratch
- attempting Anthropic-level causal interpretability

### What keeps it feasible
- emotion datasets are available
- local inference is lightweight enough
- fine-tuning pretrained audio transformers is more practical than full training from scratch
- the demo can run offline
- interpretability analysis can be done on saved embeddings after training

---

## 17. Risks and Mitigations

### Risk 1. Live demo fails on unseen voice
**Cause:** speaker mismatch or weak generalization.

**Mitigation:**
- evaluate on held-out speakers during development
- include a few personal calibration recordings if appropriate
- prepare backup recordings
- present live demo as stress test, not guaranteed perfection

### Risk 2. Model overfits acted datasets
**Cause:** datasets may be small and somewhat artificial.

**Mitigation:**
- use data augmentation
- use speaker-independent splits
- include cross-dataset or live evaluation

### Risk 3. Too much project scope
**Cause:** trying to do classification, multimodal learning, causal analysis, and deployment all at once.

**Mitigation:**
- keep the scope to audio-only
- one main model plus one baseline
- simple interpretability analysis instead of full mechanistic tracing

### Risk 4. Dataset label mismatch
**Cause:** different datasets use slightly different emotion taxonomies.

**Mitigation:**
- create a shared reduced label set
- document mapping decisions clearly

### Risk 5. Weak novelty perception
**Cause:** plain emotion classification may feel standard.

**Mitigation:**
- emphasize seminar bridge
- emphasize transformer comparison
- emphasize speaker generalization and latent emotion analysis

---

## 18. Team Division of Work

This project works well for a group because it naturally splits into sub-problems.

### Team role suggestions

#### 1. Data and preprocessing team
- collect datasets
- harmonize labels
- preprocess audio
- implement augmentation and splits

#### 2. Baseline modeling team
- build CNN baseline
- run early experiments
- produce initial benchmark tables

#### 3. Transformer modeling team
- fine-tune or build main transformer model
- tune training parameters
- save checkpoints and embeddings

#### 4. Evaluation and interpretability team
- compute metrics
- build confusion matrices
- run PCA/UMAP/t-SNE visualizations
- perform layer probing and centroid analysis

#### 5. Demo and presentation team
- build UI in Gradio or Streamlit
- prepare demo script
- prepare slides, visuals, and project narrative

In practice, some students can cover more than one role depending on group size.

---

## 19. Software and Local Setup

### Programming stack
- Python
- PyTorch
- torchaudio / librosa
- scikit-learn
- matplotlib / seaborn or plotly for analysis
- Gradio or Streamlit for demo interface

### Possible model libraries
- Hugging Face Transformers
- SpeechBrain
- timm if using spectrogram transformer style implementation

### Local hardware expectations
This project does **not** require large-scale LLM hosting.

#### For training
- a machine with a CUDA-capable GPU is ideal
- moderate GPU memory is helpful, especially for transformer fine-tuning
- smaller models or frozen-feature extraction can reduce hardware needs

#### For inference/demo
- CPU inference may be acceptable for a small model
- GPU is helpful but not mandatory for the demo

---

## 20. Deliverables

### 1. Working trained model
A baseline and a transformer-based emotion classifier.

### 2. Evaluation results
- metrics table
- confusion matrix
- held-out speaker results
- same-sentence comparison results

### 3. Interpretability analysis
- hidden representation visualizations
- layer probe results
- class centroid comparisons
- optional latent direction intervention results

### 4. Live demo
A local interface that records audio and predicts emotion probabilities.

### 5. Final presentation
15-minute class presentation showing:
- motivation
- architecture
- experiments
- results
- live demo
- future work

### 6. Final report
A structured report documenting:
- problem statement
- related work
- methods
- experiments
- discussion
- limitations
- references

---

## 21. Proposed Timeline

### Weeks 1-2
- finalize project scope
- confirm datasets
- assign team roles
- set up code repository and environment

### Weeks 3-4
- preprocessing pipeline
- baseline dataset exploration
- label harmonization
- first baseline CNN implementation

### Weeks 5-6
- train and evaluate CNN baseline
- produce initial metrics and confusion matrix

### Weeks 7-9
- implement or fine-tune transformer-based model
- run main experiments
- compare against baseline

### Weeks 10-11
- held-out speaker testing
- same-sentence controlled tests
- record sample demo audio

### Weeks 12-13
- interpretability sub-study
- extract hidden states
- run visualization and probing analyses

### Weeks 14-15
- build live demo interface
- integrate final model
- prepare presentation materials

### Final week
- dry run demo
- finalize report
- present project

---

## 22. Evaluation Criteria for Success

The project should define success in multiple dimensions.

### Technical success
- model produces sensible emotion predictions
- transformer model is at least competitive with the baseline
- system runs locally for demo

### Research success
- speaker-independent evaluation is completed
- same-sentence emotional variation test is demonstrated
- latent representation analysis reveals meaningful structure or at least an interpretable negative result

### Presentation success
- the audience understands the seminar-to-project bridge
- the live demo is clear and memorable
- limitations are discussed honestly

---

## 23. Ethical and Scientific Cautions

Emotion recognition is not perfect or universal. Emotional expression depends on:

- speaker identity
- culture
- accent
- recording conditions
- acted vs natural speech
- annotation subjectivity

Because of this, the project should clearly state that:

- the model predicts labels from available datasets
- predictions are probabilistic and imperfect
- output should not be interpreted as definitive psychological truth
- the interpretability sub-study concerns model representations, not human emotions

---

## 24. Recommended Final Project Title Options

### Option 1
**Beyond LLMs: Transformer-Based Speech Emotion Recognition with Latent Emotion Representation Analysis**

### Option 2
**Speaker-Independent Speech Emotion Recognition Using Audio Transformers**

### Option 3
**Audio Emotion Recognition with Transformer Models and Emotion-Direction Analysis**

### Option 4
**From Transformers Beyond LLMs to Speech Emotion Recognition: Audio Modeling and Latent Emotion Analysis**

---

## 25. Final Recommendation

This is a strong project idea if it is framed properly.

The best version is **not** just:

> "We built an app that detects anger from audio."

The best version is:

> "We built and evaluated a transformer-based speech emotion recognition system, compared it with a baseline, studied same-sentence emotional variation and speaker generalization, and explored whether the model's hidden representations contain interpretable emotion structure inspired by current emotion-vector research."

That framing makes the project:

- relevant to the course
- aligned with the seminar theme
- technically substantial
- presentation-friendly
- current and research-aware
- feasible within a semester if scoped well

---

## 26. Suggested References to Build From

1. Anthropic research on emotion concepts and internal emotion vectors in Claude.
2. wav2vec 2.0 for self-supervised speech representation learning.
3. Audio Spectrogram Transformer (AST) for attention-based audio classification.
4. RAVDESS dataset documentation.
5. CREMA-D dataset documentation.
6. IEMOCAP dataset documentation.
7. Prior speech emotion recognition work using wav2vec 2.0 representations.

These references are enough to support the seminar bridge, model choice, dataset choice, and interpretability inspiration for the proposal and final report.
