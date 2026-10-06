# Speech Emotion Recognition: Feature Representation Comparison

A comparative study of speech feature representations for emotion classification, evaluating classical machine learning against deep learning approaches.

## Project Overview

This project investigates how different speech feature representations affect emotion classification performance, and compares a CNN's performance against simpler baseline models. Using the CREMA-D dataset, we explore whether deep learning on mel-spectrograms outperforms traditional approaches like SVM on handcrafted MFCC features.

## Research Question

**How do different speech feature representations affect speech emotion classification performance, and how well does a CNN perform compared with simpler baseline models?**

## Key Findings

| Model | Feature Representation | Test Accuracy | Macro F1 |
|-------|----------------------|---------------|----------|
| SVM (RBF kernel) | MFCC summary statistics (mean + std) | **53.9%** | **53.7%** |
| CNN (3 conv layers) | Mel-spectrograms | 50.4% | 50.2% |

**Result:** The classical SVM baseline slightly outperformed the CNN on this dataset, suggesting that for limited training data, well-engineered summary features can be competitive with learned representations.

## Dataset

- **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset)
- 7,442 audio clips from 91 actors
- 6 emotion classes: angry, disgust, fear, happy, neutral, sad
- Balanced distribution across most classes

## Methodology

### Pipeline Overview

```text
Audio Files → Feature Extraction → Model Training → Evaluation
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   MFCC Summary            Mel-Spectrogram
   (40 features)           (128 × 108 × 1)
        ↓                       ↓
      SVM                    CNN
```

### Feature Representations

**1. MFCC Summary Features (Baseline)**
- 20 MFCC coefficients
- Aggregated over time using mean and standard deviation
- Final feature vector: 40 dimensions
- Standardized using StandardScaler

**2. Mel-Spectrograms (CNN)**
- 128 mel frequency bands
- 2.5 second audio clips (fixed length)
- Per-sample normalization (zero mean, unit variance)
- Shape: 128 × 108 × 1 (treated as single-channel image)

### Models

**SVM Baseline**
- RBF kernel, C=3, gamma='scale'
- Scikit-learn Pipeline with StandardScaler

**CNN Architecture**
- 3 convolutional blocks (16 → 32 → 64 filters)
- MaxPooling after each block
- Dropout (0.3) before dense layer
- 64-unit dense layer → 6-class softmax output
- Adam optimizer, categorical cross-entropy loss
- Early stopping on validation loss (patience=5)

### Data Split

- Training: 70% (5,209 samples)
- Validation: 15% (1,116 samples)
- Test: 15% (1,117 samples)
- Stratified by emotion class

## Project Structure

```text
speech-emotion-recognition/
├── SER_Project_Scaffold.ipynb   # Main notebook with full pipeline
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

## Results

### Confusion Matrices

Both models showed similar confusion patterns:
- **Happy** and **fear** were often confused with each other
- **Neutral** was frequently predicted for other emotions
- **Angry** achieved the highest per-class F1 in both models (0.68 and 0.67)

### Per-Class Performance (SVM)

| Emotion | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Angry | 0.72 | 0.64 | 0.68 |
| Disgust | 0.49 | 0.41 | 0.45 |
| Fear | 0.53 | 0.42 | 0.47 |
| Happy | 0.51 | 0.55 | 0.53 |
| Neutral | 0.47 | 0.61 | 0.53 |
| Sad | 0.53 | 0.62 | 0.57 |

## Key Insights

1. **Feature engineering matters:** MFCC summary statistics proved surprisingly effective for this task, suggesting that temporal aggregation captures important emotion cues.

2. **Limited data constrains deep learning:** The CNN's performance was likely limited by dataset size (7,442 samples). Deep learning typically benefits from larger corpora.

3. **Confusion patterns are consistent:** Both models struggled with similar emotion pairs (fear/happy, disgust/neutral), indicating inherent acoustic similarity between these emotions.

4. **Angry is most distinguishable:** High energy and distinctive spectral characteristics make anger the easiest emotion to detect.

## Technologies Used

- **Python 3**
- **librosa** - Audio loading and feature extraction
- **TensorFlow/Keras** - CNN implementation
- **scikit-learn** - SVM, preprocessing, evaluation metrics
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn** - Visualization

## Getting Started

### Installation

```bash
pip install librosa tensorflow scikit-learn pandas numpy matplotlib seaborn tqdm kagglehub
```

### Running the Notebook

1. Open `SER_Project_Scaffold.ipynb` in Jupyter or Google Colab
2. The notebook automatically downloads CREMA-D via `kagglehub`
3. Run cells sequentially to reproduce results

### Configuration

Key parameters can be adjusted in the notebook:
- `duration=2.5` - Audio clip length in seconds
- `n_mfcc=20` - Number of MFCC coefficients
- `n_mels=128` - Number of mel frequency bands
- `epochs=25` - Maximum training epochs for CNN

## Future Improvements

- [ ] Add Random Forest baseline with prosodic features (pitch, energy, speaking rate)
- [ ] Experiment with data augmentation (time stretching, pitch shifting, noise injection)
- [ ] Try transfer learning from pre-trained audio models (VGGish, YAMNet)
- [ ] Implement cross-validation for more robust evaluation
- [ ] Explore attention mechanisms for temporal modeling

## Authors

Cis Garcia and Pranav Nallaperumal

## Acknowledgments

- CREMA-D dataset: Cao et al., "CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset"
- Kaggle dataset hosting: ejlok1/cremad
