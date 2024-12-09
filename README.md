Early Detection of Alzheimer’s Disease Through Speech Analysis

Overview

This project explores the use of speech analysis to predict early signs of Alzheimer’s disease (AD) by leveraging both linguistic and acoustic features. Using the DementiaBank Pitt Corpus, we developed a multi-branch neural network that integrates audio and text features to classify speech samples as Alzheimer’s or control with high accuracy and interpretability. This project highlights the potential of non-invasive, speech-based diagnostic tools in healthcare.

Features

Multi-Branch Neural Network:
Processes audio and text features in separate branches and combines them for joint analysis.
Linguistic Feature Extraction:
Extracts features such as Total Words, Unique Words, Mean Sentence Length, and Disfluencies.
Model Evaluation:
High-performance metrics: Accuracy (97.5%), ROC-AUC (98%), and robust recall for early detection.
Data Visualization:
Correlation matrices and scatter plots for feature insights.
Confusion matrix and training curves for model performance analysis.
Dataset

The project uses the DementiaBank Pitt Corpus, a publicly available dataset containing:

Speech Recordings: Audio files of participants describing pictures or engaging in spontaneous speech tasks.
Transcripts: Text transcriptions paired with the audio recordings.
Access the dataset at DementiaBank.

Installation

1. Clone the Repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
2. Set Up the Environment
Create and activate a virtual environment:

python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
Install the dependencies:

pip install -r requirements.txt

Usage

Data Preprocessing:
Download the DementiaBank Pitt Corpus and preprocess the data using the provided scripts.
Ensure the dataset is organized into data/processed/X_scaled.csv and data/processed/y.csv.
Training the Model:
Run the script to train the multi-branch neural network:
python models/multi_branch_nn.py
Evaluating the Model:
Evaluate the trained model on the test set and visualize performance metrics:
python models/evaluate_model.py
Project Structure

.
├── data/
│   ├── raw/                  # Original data files
│   ├── processed/            # Preprocessed data
├── models/
│   ├── multi_branch_nn.py    # Neural network architecture and training
├── notebooks/
│   ├── EDA/                  # Plots and graphs, regressions, feature detection etc.
├── README.md                 # Project overview

Results

Performance Metrics:
Accuracy: 97.5%
ROC-AUC: 98%
Precision: 95%
Recall: 96%
Feature Insights:
Significant predictors: Mean Sentence Length and Disfluencies.
Correlation matrix highlights redundancy between Total Words and Unique Words.
Visualizations:
Loss and accuracy curves show consistent training progress.
Confusion matrix indicates minimal misclassifications.

Future Work

Extend the dataset with more diverse speech samples.
Incorporate additional acoustic features such as pitch, tone, and pauses.
Experiment with advanced architectures like transformers or ensemble models.
Explore longitudinal data for tracking changes in speech over time.
Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request for improvements or additional features.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Contact

For questions or feedback, reach out to:

Your Name: adam.rencher12@gmail.com
LinkedIn Profile: linkedin.com/in/adam-rencher-8294381a5
GitHub Profile: adarench
