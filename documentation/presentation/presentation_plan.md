# Presentation Plan: Causal Emotion Geometry in a Fine-Tuned Speech Model

## Purpose

This file is the review draft for the final project presentation. It is not the slide deck yet. After review, this outline can be converted into a PowerPoint/Google Slides deck.

The presentation should tell one clear story:

> We started with speech emotion recognition, but the real project is about whether a trained speech model contains manipulable internal emotion directions that help cause its output.

The safest final claim is:

> The model does not have emotions as experiences. It has internal emotion-like representations that are predictive, causal, steerable, removable, and transferable through activation patching.

## Timing And Structure

Target presentation length: 30 minutes.

Recommended slide count: 25 main slides, plus optional backup slides.

Speaker split: 5 speakers, 5 sections, about 6 minutes each.

| Speaker | Section | Time | Main Job |
|---|---:|---:|---|
| Speaker 1 | Motivation and Anthropic bridge | 5.5-6 min | Explain why this is more than ordinary emotion classification |
| Speaker 2 | Dataset, model, and baseline | 5.5-6 min | Establish that the supervised model is real and evaluated fairly |
| Speaker 3 | Emotion directions and predictive geometry | 5.5-6 min | Show that emotion exists as a readable hidden-space direction |
| Speaker 4 | Causal interventions | 6-6.5 min | Show that changing directions changes model behavior |
| Speaker 5 | Activation patching, robustness, and conclusion | 6-6.5 min | Deliver the strongest causal result and the final interpretation |

Expected pacing: about 60-75 seconds per normal slide, 90 seconds for complex result slides.

## Design Direction

Use a clean research-presentation style rather than a flashy app-demo style.

Recommended visual tone:

- Dark charcoal or warm off-white background.
- One accent color for the "emotion vector" idea, such as amber or teal.
- Use diagrams and large numbers more than paragraphs.
- Avoid crowded screenshots of notebooks.
- Put one claim per slide.
- Use figure captions as spoken explanation, not as tiny slide text.

Suggested recurring visual metaphor:

> Model behavior is not just a classifier head; it is shaped by directions in hidden-state space.

## Slide Visual Map

Use this table when building the actual deck. "Slide-native" means we should create a clean diagram directly in PowerPoint/Google Slides instead of pasting a notebook plot. That will make the deck feel more polished and easier to present.

| Slide | Main Visual/Image To Use | Notes |
|---:|---|---|
| 1 | Slide-native title visual: waveform fading into an arrow/vector field | No notebook plot; make this clean and cinematic. |
| 2 | Slide-native comparison diagram: `audio -> classifier -> label` vs `audio -> hidden geometry -> probability` | Use this to introduce prediction vs mechanism. |
| 3 | Slide-native two-column Anthropic-to-our-project bridge | Show LLM text vectors on left, speech emotion directions on right. |
| 4 | Slide-native three-test diagram: `Read`, `Steer`, `Patch` | Use three icons/cards. |
| 5 | Slide-native roadmap/timeline | Six-step horizontal flow from training to patching. |
| 6 | `documentation/assets/figures/01_eda_cell_10_fig_02_final_6_class_label_counts.png` | Use for final six-class RAVDESS distribution. |
| 7 | Slide-native speaker split diagram | Actor blocks: train `01-16`, validation `17-20`, test `21-24`. |
| 8 | Slide-native model pipeline diagram | `Audio -> wav2vec2 -> pooled hidden state -> classifier head -> probabilities`. |
| 9 | `documentation/assets/figures/06_model_comparison_cell_04_fig_01_model_comparison.png` | Use for wav2vec2 vs CNN performance. |
| 10 | Slide-native hidden-state extraction diagram | Show trained model producing embeddings for analysis. |
| 11 | Slide-native centroid equation visual | Show `d_emotion = centroid_emotion - centroid_neutral`. |
| 12 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_09_fig_06_anthropic_style_emotion_vector_analysis.png` | PCA centroid and neutral-to-emotion direction view. |
| 13 | `documentation/assets/figures/08_direction_only_classification_cell_06_fig_01_direction_only_vs_trained_classifier.png` | Direction-only vs trained classifier per-class F1. |
| 14 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_05_fig_02_anthropic_style_emotion_vector_analysis.png` | Projection vs output probability scatter panels. |
| 15 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_08_fig_05_anthropic_style_emotion_vector_analysis.png` | Use if readable for intensity/same-context summary; otherwise make a slide-native paired-clip diagram with the cosine range. |
| 16 | Slide-native decodability vs causal-use diagram | Two cards: "can read it" vs "model uses it". |
| 17 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_07_fig_04_anthropic_style_emotion_vector_analysis.png` | Use for positive/negative steering if readable; otherwise make a cleaner before/after probability bar. |
| 18 | `documentation/assets/figures/09_causal_ablation_study_cell_06_fig_01_overall_ablation_results.png` | Final-embedding direction ablation result. |
| 19 | `documentation/assets/figures/09_causal_ablation_study_cell_12_fig_04_ablation_strength_sweep.png` | Ablation dose-response sweep. |
| 20 | `documentation/assets/figures/14_mid_transformer_causal_intervention_cell_13_fig_03_3b_norm_matched_random_vector_sufficiency_control.png` | Random-vector control; can also use this later if Slide 21 needs more space. |
| 21 | `documentation/assets/figures/14_mid_transformer_causal_intervention_cell_11_fig_02_3_experiment_1_mdash_mid_transformer_sufficiency_sweep.png` | Mid-transformer sufficiency by layer. |
| 22 | Slide-native activation patching diagram | Show source hidden state patched into destination model stream. |
| 23 | `documentation/assets/figures/15_activation_patching_cell_13_fig_01_4_experiment_1_mdash_layer_wise_patch_transfer.png` | Main layer-wise patch transfer result. |
| 24 | `documentation/assets/figures/15_activation_patching_cell_27_fig_07_9_experiment_6_mdash_patch_controls_and_direction_component_decomposition.png` | Patch controls and direction-component decomposition. |
| 25 | `documentation/assets/figures/16_robustness_and_negative_controls_cell_13_fig_02_5_visual_summary.png` plus slide-native final claim box | Use the robustness visual only if it stays readable; otherwise use the final claim and limitation cards. |

Optional backup visuals:

| Backup Slide | Main Visual/Image To Use | Notes |
|---|---|---|
| Backup A | Slide-native notebook map | Show notebooks grouped by project stage. |
| Backup B | `documentation/assets/figures/11_cross_dataset_transfer_cell_11_fig_01_transfer_evaluation.png` | Use for CREMA-D transfer boundary. |
| Backup C | `documentation/assets/figures/13_sparse_autoencoder_cell_10_fig_03_feature_emotion_heatmap.png` | Use for SAE exploratory result if needed. |

## Section 1: Motivation And Anthropic Bridge

Owner: Speaker 1

Target time: 5.5-6 minutes

Goal: Make the audience understand the project question before showing any numbers.

### Slide 1: Title

Time: 30 seconds

Title: `Causal Emotion Geometry in a Fine-Tuned Speech Model`

Subtitle: `An Anthropic-inspired study of direction steering and activation patching`

Content:

- Course final project.
- Team member names.
- One-line tagline: `Emotion in this speech model is not a feeling; it is a causal geometry in hidden state space.`

Speaker note:

Open by saying this began as a speech emotion recognition project, but became a model-interpretability project.

### Slide 2: The Normal SER Question Is Not Enough

Time: 60 seconds

Main claim:

Most speech emotion recognition projects stop at: "Can we classify angry, happy, sad, etc.?"

But our question is deeper:

> After the model learns the task, where is emotion represented internally, and does that representation affect the output?

Visual suggestion:

- Left: audio waveform -> model -> emotion label.
- Right: audio waveform -> model hidden space -> emotion direction -> output probability.

Speaker note:

Explain the difference between prediction and mechanism. We care about what internal structure the model uses to produce the prediction.

### Slide 3: Anthropic Inspiration

Time: 75 seconds

Main claim:

Anthropic studied emotion concept vectors inside a language model and showed that internal vectors could correlate with, and influence, generated behavior.

Translate carefully:

- Anthropic model: text stories/dialogues -> internal emotion vectors -> changed language behavior.
- Our model: emotional speech clips -> internal emotion directions -> changed class probabilities.

Visual suggestion:

Two-column bridge diagram:

| Anthropic LLM | Our Speech Model |
|---|---|
| Text emotion contexts | RAVDESS acted emotional speech |
| Residual-stream concept vectors | wav2vec2 hidden-state directions |
| Steering changes generated behavior | Steering/patching changes predicted emotion |

Speaker note:

Do not say "models feel emotions." Say "models can represent emotion concepts functionally."

### Slide 4: Core Research Question

Time: 60 seconds

Question:

> Does a speech emotion model only learn labels, or does it learn internal emotion directions that causally shape the final prediction?

Hypotheses:

- If directions are real, projection onto them should correlate with emotion probabilities.
- If directions are causal, adding/removing them should change predictions.
- If hidden states carry emotion identity, patching a hidden state from one clip into another should transfer emotion evidence.

Visual suggestion:

Three test icons:

- Read: projection/correlation.
- Steer: add/remove vector.
- Patch: swap hidden state.

### Slide 5: Roadmap

Time: 60 seconds

Main flow:

1. Train a speaker-independent wav2vec2 emotion classifier.
2. Build neutral-to-emotion directions in hidden space.
3. Test whether those directions predict probabilities.
4. Intervene with steering, ablation, and mid-transformer injection.
5. Patch real hidden states between matched clips.
6. Add controls and limits.

Transition to Speaker 2:

> Before we can interpret the model, we first need a real model trained and evaluated under a fair split.

## Section 2: Dataset, Model, And Baseline

Owner: Speaker 2

Target time: 5.5-6 minutes

Goal: Show the foundation is solid: dataset, split, model, and baseline.

### Slide 6: Dataset: RAVDESS Speech

Time: 60 seconds

Main content:

- Dataset: RAVDESS audio-only speech.
- Original speech clips: 1,440.
- Final task: 1,248 clips after mapping calm into neutral and dropping surprised.
- Classes: neutral, happy, sad, angry, fearful, disgust.

Suggested visual:

Use `documentation/assets/figures/01_eda_cell_10_fig_02_final_6_class_label_counts.png`

Speaker note:

Mention RAVDESS is acted speech, which is a strength for controlled comparisons but a limitation for real-world emotion.

### Slide 7: Speaker-Independent Split

Time: 60 seconds

Main content:

- Train actors: 01-16.
- Validation actors: 17-20.
- Test actors: 21-24.

Main claim:

The model is tested on speakers it did not see during training.

Visual suggestion:

Simple split diagram with actor blocks.

Speaker note:

This matters because otherwise emotion recognition can become speaker memorization.

### Slide 8: Model Choice

Time: 75 seconds

Main content:

- Main model: `facebook/wav2vec2-base`.
- It is a pretrained raw-speech encoder.
- We fine-tuned it for six-way emotion classification.
- Baseline: CNN on log-mel spectrograms.

Visual suggestion:

Pipeline diagram:

`Audio -> wav2vec2 encoder -> pooled hidden state -> classification head -> emotion probabilities`

Speaker note:

The baseline helps show that performance is not trivial. wav2vec2 uses pretrained speech representations, while the CNN is a simpler acoustic baseline.

### Slide 9: Supervised Performance

Time: 75 seconds

Key numbers:

- wav2vec2 test accuracy: `0.8269`.
- wav2vec2 test macro F1: `0.8126`.
- CNN test accuracy: `0.4231`.
- CNN test macro F1: `0.3241`.

Suggested visual:

Use `documentation/assets/figures/06_model_comparison_cell_04_fig_01_model_comparison.png`

Speaker note:

Say the purpose is not to maximize leaderboard accuracy. The purpose is to get a strong enough model that its internals are worth interpreting.

### Slide 10: Why This Model Is Interpretable Enough

Time: 60 seconds

Main claim:

Once wav2vec2 performs well on held-out speakers, we can ask what its hidden states contain.

Bridge:

- We extract hidden embeddings.
- We compare class centroids.
- We define directions like `angry - neutral`.
- We test whether those directions predict and cause outputs.

Transition to Speaker 3:

> Now that we have a working speech model, the next step is to ask whether emotion is organized geometrically inside it.

## Section 3: Emotion Directions And Predictive Geometry

Owner: Speaker 3

Target time: 5.5-6 minutes

Goal: Explain how emotion directions are built and why they are meaningful before causal tests.

### Slide 11: Defining Emotion Directions

Time: 75 seconds

Core equation:

```text
emotion direction = mean_embedding(emotion) - mean_embedding(neutral)
```

Examples:

- `happy direction = happy centroid - neutral centroid`
- `angry direction = angry centroid - neutral centroid`

Important control:

We also center embeddings by actor and statement so directions are less likely to just encode speaker or sentence.

Visual suggestion:

Centroid arrow diagram from neutral to emotion.

### Slide 12: PCA View Of Emotion Geometry

Time: 75 seconds

Main claim:

Emotion classes form structured regions in hidden space, and neutral-to-emotion arrows point toward emotion-specific clusters.

Suggested visual:

Use `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_09_fig_06_anthropic_style_emotion_vector_analysis.png`

Speaker note:

This figure is intuition, not proof. PCA is a 2D projection, so the real evidence comes next.

### Slide 13: Direction-Only Classifier

Time: 75 seconds

Main claim:

A simple classifier using only projections onto emotion directions nearly matches the trained classifier head.

Key numbers:

- Controlled direction-only accuracy: `0.8413`.
- Controlled direction-only macro F1: `0.8301`.
- Trained head macro F1 in embedding analysis: `0.8237`.
- Shuffled-label directions macro F1: `0.0151`.

Suggested visual:

Use `documentation/assets/figures/08_direction_only_classification_cell_06_fig_01_direction_only_vs_trained_classifier.png`

Speaker note:

This is a major result: the hidden space is not just a black-box cloud. A small set of directions recovers the model's behavior.

### Slide 14: Projection Tracks Probability

Time: 75 seconds

Main claim:

If a clip projects strongly onto the angry direction, the model tends to assign higher angry probability.

Key numbers:

- Test-set Pearson correlations across emotions: about `0.68` to `0.77`.
- True-class subset correlations: up to `0.9525`.

Suggested visual:

Use `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_05_fig_02_anthropic_style_emotion_vector_analysis.png`

Speaker note:

This turns the vector from a visualization into a behavior-linked variable.

### Slide 15: Same-Context And Intensity Checks

Time: 60 seconds

Main claim:

RAVDESS lets us compare clips with the same actor and sentence but different emotion.

Key results:

- Same-context displacement points in the expected emotion direction.
- Mean cosine to expected direction: `0.8203` to `0.9686`.
- Strong-intensity clips generally project farther along the same direction than normal-intensity clips.

Transition to Speaker 4:

> So far, the directions are readable and predictive. But causal interpretability requires a harder test: what happens when we change them?

## Section 4: Causal Interventions

Owner: Speaker 4

Target time: 6-6.5 minutes

Goal: Show that emotion directions are not just readable; changing them changes model behavior.

### Slide 16: From Decodability To Causality

Time: 60 seconds

Main distinction:

- Decodability: a direction contains information.
- Causal use: changing the direction changes the output.

Tests in this section:

- Add direction.
- Subtract direction.
- Remove direction components.
- Inject directions inside transformer layers.
- Compare against random controls.

Speaker note:

This slide is important because it explains why we did more than classification.

### Slide 17: Steering: Adding Emotion Directions

Time: 75 seconds

Main claim:

Adding an emotion direction increases the model's probability for that emotion.

Key number:

- Alpha `0.5` mean target probability delta: `+0.1921`.
- 95% CI: `[0.1800, 0.2043]`.

Speaker note:

Also mention negative steering: subtracting the direction lowers the target probability, which strengthens the causal claim.

Visual suggestion:

Use a simple before/after probability bar or the steering figure from notebook 07 if making the final deck.

### Slide 18: Ablation: Removing Directions Hurts

Time: 75 seconds

Main claim:

Removing the emotion-direction component damages classification.

Key numbers:

- Macro F1 drops from `0.8237` to `0.5348`.
- Delta: `-0.2889`.
- 95% CI: `[-0.3395, -0.2487]`.

Suggested visual:

Use `documentation/assets/figures/09_causal_ablation_study_cell_06_fig_01_overall_ablation_results.png`

Speaker note:

This is a necessity test: if removing the direction breaks performance, the direction is load-bearing.

### Slide 19: Ablation Strength Sweep

Time: 60 seconds

Main claim:

Performance stays stable under partial removal but drops sharply when the direction component is fully removed.

Suggested visual:

Use `documentation/assets/figures/09_causal_ablation_study_cell_12_fig_04_ablation_strength_sweep.png`

Speaker note:

This dose-response pattern makes the ablation result more convincing than a single before/after number.

### Slide 20: Random Direction Control

Time: 60 seconds

Main claim:

Norm-matched random directions do not reproduce the same damage.

Key numbers:

- Random ablation null mean: `0.0002`.
- 95% CI: `[0.0000, 0.0038]`.
- Fraction as damaging as real ablation: `0.0`.

Speaker note:

This answers the criticism: "Maybe any vector perturbation would change outputs." It does not.

### Slide 21: Mid-Transformer Injection

Time: 90 seconds

Main claim:

Emotion directions are causal not only at the final embedding, but inside wav2vec2 during the forward pass.

Key numbers:

- Strongest sufficiency layer: layer `7`.
- Mean target probability delta at alpha `0.5`: `+0.2545`.
- Most damaging necessity layer: layer `8`, macro-F1 delta `-0.1513`.

Suggested visual:

Use `documentation/assets/figures/14_mid_transformer_causal_intervention_cell_11_fig_02_3_experiment_1_mdash_mid_transformer_sufficiency_sweep.png`

Speaker note:

This is stronger than post-hoc steering because the model must process the injected direction through the remaining transformer layers.

Transition to Speaker 5:

> We have shown synthetic vector interventions. The final question is whether real hidden states from real clips carry transferable emotion evidence.

## Section 5: Activation Patching, Robustness, And Final Claim

Owner: Speaker 5

Target time: 6-6.5 minutes

Goal: Present the strongest result, handle limitations honestly, and land the final claim.

### Slide 22: Activation Patching Setup

Time: 75 seconds

Main idea:

Take a destination clip and replace its hidden state at a selected layer with the hidden state from a source clip.

Controlled pair:

- Same actor.
- Same sentence.
- Same repetition.
- Same intensity.
- Different emotion.

Interpretation:

If the destination prediction moves toward the source emotion, the patched hidden state carries emotion identity.

Visual suggestion:

Diagram:

`source angry clip hidden state -> patch into destination neutral clip -> output shifts toward angry`

### Slide 23: Activation Patching Result

Time: 90 seconds

Main claim:

Real hidden-state patches transfer emotion identity.

Key numbers:

- Same-context ordered pairs: `640`.
- Best layer: `11`.
- Mean source-emotion probability delta: `+0.6796`.
- 95% CI: `[0.6475, 0.7092]`.
- Prediction flips to source emotion in `75.62%` of pairs.

Suggested visual:

Use `documentation/assets/figures/15_activation_patching_cell_13_fig_01_4_experiment_1_mdash_layer_wise_patch_transfer.png`

Speaker note:

This is the most concrete causal analogue to the Anthropic-style idea: a hidden state from one emotional context can alter another output.

### Slide 24: Patch Controls And Direction Component

Time: 90 seconds

Main claim:

Most of the patch effect is carried by the emotion-direction component.

Key numbers:

- Full same-context patch: `+0.6796`, flip `0.7562`.
- Direction-component patch: `+0.6395`, flip `0.7266`.
- Orthogonal residual: `+0.0533`, flip `0.1094`.
- Self-patch and random-source controls mostly fail.

Suggested visual:

Use `documentation/assets/figures/15_activation_patching_cell_27_fig_07_9_experiment_6_mdash_patch_controls_and_direction_component_decomposition.png`

Speaker note:

This connects the real hidden-state patching result back to the emotion-vector story.

### Slide 25: Robustness, Boundaries, And Final Answer

Time: 90 seconds

Robustness:

- Bootstrap CIs support main effects.
- Per-test-actor metrics show no single actor drives the result.
- Shuffled/random controls fail.

Boundaries:

- RAVDESS is acted speech, not spontaneous emotion.
- Directions are learned from labels, not discovered completely unsupervised.
- CREMA-D transfer is weak, so geometry is domain-sensitive.
- Activation patching is not a fully isolated circuit, although direction-component decomposition helps.

Final answer:

> No, the model does not have emotions as experiences.
>
> Yes, it has manipulable internal representations corresponding to emotion concepts.

Close:

> Emotion in this speech model is not a feeling; it is a causal geometry in hidden state space.

## Optional Backup Slides

These should not be in the 30-minute main flow unless the instructor asks.

### Backup A: Full Notebook Map

Purpose:

Show that the project is reproducible and each notebook has a role.

Content:

- `01`: EDA and dataset construction.
- `02`: wav2vec2 fine-tuning.
- `05`: CNN baseline.
- `07`: Anthropic-style vectors.
- `09`: ablation.
- `14`: mid-transformer intervention.
- `15`: activation patching.
- `16`: robustness and negative controls.

### Backup B: Cross-Dataset Transfer

Main result:

CREMA-D transfer is weak.

Numbers:

- trained-head macro F1: `0.3498`.
- centroid macro F1: `0.3473`.
- direction-only macro F1: `0.3444`.

Interpretation:

The geometry is strong inside RAVDESS-style speech, but not universal across datasets.

### Backup C: Sparse Autoencoder Experiment

Main result:

The SAE is exploratory.

Numbers:

- normalized MSE: `0.1363`.
- mean cosine: `0.9390`.
- classifier macro F1 preserved: `0.8237 -> 0.8222`.

Interpretation:

Classifier-relevant information survives the bottleneck, but we should not overclaim monosemantic emotion units.

## Figure Asset Checklist

Use these figures in the final deck unless we decide to redesign them as cleaner slide-native charts.

| Slide | Figure |
|---:|---|
| 6 | `documentation/assets/figures/01_eda_cell_10_fig_02_final_6_class_label_counts.png` |
| 9 | `documentation/assets/figures/06_model_comparison_cell_04_fig_01_model_comparison.png` |
| 12 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_09_fig_06_anthropic_style_emotion_vector_analysis.png` |
| 13 | `documentation/assets/figures/08_direction_only_classification_cell_06_fig_01_direction_only_vs_trained_classifier.png` |
| 14 | `documentation/assets/figures/07_anthropic_style_emotion_vectors_cell_05_fig_02_anthropic_style_emotion_vector_analysis.png` |
| 18 | `documentation/assets/figures/09_causal_ablation_study_cell_06_fig_01_overall_ablation_results.png` |
| 19 | `documentation/assets/figures/09_causal_ablation_study_cell_12_fig_04_ablation_strength_sweep.png` |
| 21 | `documentation/assets/figures/14_mid_transformer_causal_intervention_cell_11_fig_02_3_experiment_1_mdash_mid_transformer_sufficiency_sweep.png` |
| 23 | `documentation/assets/figures/15_activation_patching_cell_13_fig_01_4_experiment_1_mdash_layer_wise_patch_transfer.png` |
| 24 | `documentation/assets/figures/15_activation_patching_cell_27_fig_07_9_experiment_6_mdash_patch_controls_and_direction_component_decomposition.png` |

## Numbers Everyone Should Know

- RAVDESS final task: `1,248` clips, `6` classes.
- Speaker-independent split: train `01-16`, val `17-20`, test `21-24`.
- wav2vec2 test accuracy: `0.8269`.
- wav2vec2 test macro F1: `0.8126`.
- CNN test macro F1: `0.3241`.
- Controlled direction-only macro F1: `0.8301`.
- Shuffled-label direction macro F1: `0.0151`.
- Projection correlations: roughly `0.68-0.77`.
- Final-layer steering delta: `+0.1921`.
- Full direction ablation macro-F1 delta: `-0.2889`.
- Mid-transformer best injection layer: `7`.
- Mid-transformer target-probability delta: `+0.2545`.
- Activation patching best layer: `11`.
- Patch source-emotion probability delta: `+0.6796`.
- Patch flip rate: `75.62%`.

## Team Rehearsal Notes

The presentation should sound like one story, not five separate mini-projects.

Recommended handoff lines:

- Speaker 1 to Speaker 2: "To test this seriously, we first needed a fair speaker-independent emotion model."
- Speaker 2 to Speaker 3: "Once the model worked, we asked whether its hidden space had readable emotion directions."
- Speaker 3 to Speaker 4: "Readable is not enough, so we next changed those directions and measured the output."
- Speaker 4 to Speaker 5: "Synthetic interventions worked, but the strongest test is patching real hidden states from real clips."

Avoid these overclaims:

- Do not say the model "feels" emotion.
- Do not say the directions are discovered zero-shot.
- Do not say the geometry is universal across all speech datasets.
- Do not say activation patching isolates a complete circuit.

Use these precise claims instead:

- The model learns internal emotion-like variables.
- These variables are readable from hidden states.
- They are aligned with output probabilities.
- They are causally connected to model behavior.
- They can be steered, removed, and transferred through hidden-state interventions.
