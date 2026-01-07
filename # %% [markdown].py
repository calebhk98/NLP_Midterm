# %% [markdown]
# I. Introduction - - - - 
# Domain-specific area 
# Objectives 
# Dataset Description 
# Evaluation methodology 
# 

# %% [markdown]
# II. Implementation - - - - 
# Data Preprocessing 
# Baseline Performance 
# Comparative Classification methodology 
# Programming style 
# 

# %% [markdown]
# III. Conclusions - - 
# Performance Analysis & Comparative Discussion 
# Project Summary and Reflections 

# %%
import sys
import subprocess
import platform
import importlib.util
import os

def is_installed(package_name):
	"""Checks if a package is importable."""
	return importlib.util.find_spec(package_name) is not None

def has_nvidia_gpu():
	"""Checks for NVIDIA GPU via nvidia-smi command."""
	try:
		# -L lists devices. If this succeeds, we have an NVIDIA GPU.
		subprocess.check_output(['nvidia-smi', '-L'])
		return True
	except (FileNotFoundError, subprocess.CalledProcessError):
		return False

def install_package(command):
	print(f"Executing: pip install {command}")
	subprocess.check_call([sys.executable, "-m", "pip", "install"] + command.split())

print("--- Starting Environment Setup ---")

# 1. Install standard libraries
# We install these first to ensure the basic environment is ready.
base_pkgs = "datasets matplotlib pandas seaborn scikit-learn nltk transformers"
install_package(base_pkgs)

# 2. Smart PyTorch Install
if not is_installed("torch"):
	print("PyTorch not found. Analyzing hardware...")
	
	system_os = platform.system()
	
	if system_os == "Darwin":
		print("macOS detected. Installing standard PyTorch (MPS supported)...")
		install_package("torch torchvision torchaudio accelerate")
		
	elif has_nvidia_gpu():
		print(f"NVIDIA GPU detected on {system_os}. Installing CUDA-enabled PyTorch...")
		# CUDA 12.4 is the stable target for modern GPUs like your 3090
		install_package("torch torchvision torchaudio accelerate --index-url https://download.pytorch.org/whl/cu124")
		
	else:
		print(f"No dedicated NVIDIA GPU detected on {system_os}. Installing CPU-optimized PyTorch...")
		install_package("torch torchvision torchaudio accelerate")
else:
	print("PyTorch is already installed. Skipping re-installation.")

# 3. Imports 
import torch
import nltk
from transformers import DistilBertTokenizer
# (Add your other specific imports here as needed)

# Download necessary NLTK data safely
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Final Verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\n" + "="*40)
print(f"SETUP COMPLETE.")
print(f"Device Available: {str(device).upper()}") # Fixed the .upper() bug
if device.type == 'cuda':
	print(f"GPU Name: {torch.cuda.get_device_name(0)}")
	vram = torch.cuda.get_device_properties(0).total_memory / 1e9
	print(f"VRAM: {vram:.2f} GB")
print("="*40)

# %%
# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
	print("✅ Success! CUDA is active.")
	print(f"GPU: {torch.cuda.get_device_name(0)}")
	
	# Check memory usage
	total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
	print(f"Total VRAM: {total_mem:.2f} GB")
	
	# create a small tensor and move it to GPU to test
	x = torch.rand(5, 3).to("cuda")
	print("\nTest Tensor on GPU:\n", x)
else:
	print("❌ Something went wrong. Still showing CPU.")

# %%
%pip install datasets matplotlib pandas seaborn scikit-learn nltk transformers torch accelerate

from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
import os

print("Installed and imported modules")

# %%
dataset = load_dataset("imdb")

# %%
dataset

# %%

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# %%
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('punkt_tab')

# %% [markdown]
# Ok, let's look at this dataset.

# %%
print("Train size:", len(train_df))
print("Test size:", len(test_df))
print("\nClass distribution in training set:")
print(train_df["label"].value_counts())
print("\nClass distribution in test set:")
print(test_df["label"].value_counts())


print("\nReview #1:")
print(train_df.iloc[0]["text"][:500])
print("\nLabel:", train_df.iloc[0]["label"], "(0=negative, 1=positive)")

# %%
def preprocess_text(text):
	"""
	Preprocesses text for NLP tasks
	Steps: lowercase, remove HTML/special chars, tokenize, remove stopwords
	"""
	# Lowercase
	text = text.lower()
	
	# Remove HTML tags
	text = re.sub(r"<.*?>", "", text)
	
	# Remove special characters and digits
	text = re.sub(r"[^a-zA-Z\s]", "", text)
	
	# Tokenization
	tokens = word_tokenize(text)
	
	# Remove stopwords using explicit loop
	stop_words = set(stopwords.words("english"))
	filtered_tokens = []
	for word in tokens:
		if word not in stop_words:
			filtered_tokens.append(word)
	
	return " ".join(filtered_tokens)

# Test the function on one review
sample_text = train_df.iloc[0]["text"]
print("Original text (first 200 chars):")
print(sample_text[:200])
print("\nCleaned text (first 200 chars):")
print(preprocess_text(sample_text)[:200])

# %%
print("Preprocessing train data. Wait  17 seconds.")
train_df["cleaned_text"] = train_df["text"].apply(preprocess_text)
print("Train data preprocessed")

print("\nPreprocessing test data. Wait another 17 seconds.")
test_df["cleaned_text"] = test_df["text"].apply(preprocess_text)
print("Test data preprocessed")

# %% [markdown]
# Let's get a baseline. We could say a baseline is 50%, as it is evenly split, but that might be disingenuous.  So I'll do a bag of words, which while looking for a baseline, was in the results several times. 

# %%
print("Starting, wait 5 seconds.")
# CountVectorizer creates simple word count features
# max_features=5000: Only use the 5000 most common words
bagOfWordsVectorizer = CountVectorizer(max_features=5000)

# Fit on training data and transform both train and test
X_train_bow = bagOfWordsVectorizer.fit_transform(train_df["cleaned_text"])
X_test_bow = bagOfWordsVectorizer.transform(test_df["cleaned_text"])

y_train = train_df["label"]
y_test = test_df["label"]

print("Shape:", X_train_bow.shape)

# Train Logistic Regression
baseline_model = LogisticRegression(max_iter=1000, random_state=147)
baseline_model.fit(X_train_bow, y_train)

# Make predictions
baseline_predictions = baseline_model.predict(X_test_bow)

# Calculate metrics
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
baseline_precision = precision_score(y_test, baseline_predictions)
baseline_recall = recall_score(y_test, baseline_predictions)
baseline_f1 = f1_score(y_test, baseline_predictions)

print("\nBaseline:")
print("Accuracy: ", round(baseline_accuracy, 4))
print("Precision:", round(baseline_precision, 4))
print("Recall:   ", round(baseline_recall, 4))
print("F1-Score: ", round(baseline_f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, baseline_predictions, target_names=["Negative", "Positive"]))

# Save confusion matrix for later visualization
baseline_cm = confusion_matrix(y_test, baseline_predictions)
baseline_cm

# %% [markdown]
# Ok, so about ~85%, much better than 50%. 

# %%

# What: TF-IDF converts text to numbers for machine learning
#       - TF (Term Frequency): How often a word appears in a document
#       - IDF (Inverse Document Frequency): How rare/important a word is
#       - Creates 5000 features (most important words/bigrams)

print("Creating TF-IDF features, wait 13 seconds")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit on training data and transform both train and test
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df["cleaned_text"])
X_test_tfidf = tfidf_vectorizer.transform(test_df["cleaned_text"])

y_train = train_df["label"]
y_test = test_df["label"]

print("TF-IDF matrix shape:", X_train_tfidf.shape)
print("Number of features:", X_train_tfidf.shape[1])

# %%
# =============================================================================
# CELL 8: Train Naive Bayes Classifier
# Section: Comparative Classification (Section 7) - Statistical Model
# What: Naive Bayes is a probabilistic classifier
#       Works well with text data and TF-IDF features
#       Fast to train and interpret
# =============================================================================

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# Make predictions
nb_predictions = nb_model.predict(X_test_tfidf)

# Calculate metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

print("\nSTATISTICAL MODEL (Naive Bayes + TF-IDF)")
print("Accuracy: ", round(nb_accuracy, 4))
print("Precision:", round(nb_precision, 4))
print("Recall:   ", round(nb_recall, 4))
print("F1-Score: ", round(nb_f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, nb_predictions, target_names=["Negative", "Positive"]))

# Save confusion matrix for later
nb_cm = confusion_matrix(y_test, nb_predictions)

# %%
import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version Compiled:", torch.version.cuda)

# %%
# =============================================================================
# CELL 10: Setup BERT model and tokenizer
# Section: Comparative Classification (Section 7) - Embedding Model
# What: Loading DistilBERT (smaller, faster version of BERT)
#       - BERT uses word embeddings (dense vector representations)
#       - Pre-trained on massive text corpus
#       - We'll fine-tune it on our sentiment data
# =============================================================================

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
	print("GPU detected! Training will be faster.")
else:
	print("No GPU detected. Training will be slower.")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
print("Tokenizer loaded")

# %%
# =============================================================================
# CELL 11: Prepare dataset for BERT
# Section: Comparative Classification (Section 7) - Embedding Model
# What: Converting our data to format BERT expects
#       Using smaller subset (5000 train, 2500 test) for speed
#       Remove these lines if you have GPU and want to use full dataset
# =============================================================================

# OPTION 1: Use smaller subset for speed (recommended for CPU)
# train_subset = train_df.sample(n=5000, random_state=147)
# test_subset = test_df.sample(n=2500, random_state=147)

# OPTION 2: Use full dataset (comment out lines above and uncomment these)
train_subset = train_df
test_subset = test_df

print("Training on", len(train_subset), "samples")
print("Testing on", len(test_subset), "samples")

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(train_subset[["text", "label"]])
test_dataset = Dataset.from_pandas(test_subset[["text", "label"]])

print("Datasets prepared")

# %%
# =============================================================================
# CELL 12: Tokenize data for BERT
# Section: Comparative Classification (Section 7) - Embedding Model
# What: BERT needs text converted to token IDs
#       Padding/truncating to max length of 512 tokens
# =============================================================================

def tokenize_function(examples):
	return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

print("Tokenizing training data. Wait 15 seconds.")
train_tokenized = train_dataset.map(tokenize_function, batched=True)
print("Tokenizing test data. Wait another 15 seconds. ")
test_tokenized = test_dataset.map(tokenize_function, batched=True)

print("Tokenization complete!")

# %%
# =============================================================================
# CELL 13: Setup and train BERT model
# Section: Comparative Classification (Section 7) - Embedding Model
# What: Fine-tuning DistilBERT on our sentiment data
#       This will take 10-30 minutes depending on your hardware
#       2 epochs = going through the entire dataset twice
# =============================================================================

def setupModel():
	# Define where to save the model
	model_save_path = "./saved_distilbert_model"
	trainOverSave=True

	# Check if we already have a trained model saved
	if os.path.exists(model_save_path) and not trainOverSave:
		print("Found saved model! Loading it instead of training...")
		model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
		model.to(device)
		print("Model loaded successfully!")
		return model


	# Load pre-trained model
	model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
	model.to(device)

	# Training configuration
	training_args = TrainingArguments(
		output_dir="./results",
		num_train_epochs=2,
		per_device_train_batch_size=16,
		per_device_eval_batch_size=16,
		warmup_steps=500,
		weight_decay=0.01,
		logging_dir="./logs",
		logging_steps=100,
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
	)

	# Create trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_tokenized,
		eval_dataset=test_tokenized,
	)

	# Train the model
	print("Starting training... (this will take a while)")
	trainer.train()
	print("Training complete!")

	# Save the trained model
	print("Saving model to", model_save_path)
	model.save_pretrained(model_save_path)
	tokenizer.save_pretrained(model_save_path)
	return model
model = setupModel()

# %%
# =============================================================================
# CELL 13: Setup and train BERT model
# Section: Comparative Classification (Section 7) - Embedding Model
# What: Fine-tuning DistilBERT on our sentiment data
# =============================================================================

from transformers import EarlyStoppingCallback 

def setupModel():
	# Define where to save the model
	model_save_path = "./saved_distilbert_model"
	trainOverSave = False 

	# Check if we already have a trained model saved
	if os.path.exists(model_save_path) and not trainOverSave:
		print("Found saved model! Loading it instead of training...")
		# We use ignore_mismatched_sizes in case we change num_labels later
		model = DistilBertForSequenceClassification.from_pretrained(model_save_path)
		model.to(device)
		print("Model loaded successfully!")
		return model

	# Load pre-trained model (resetting weights)
	print(f"Initializing model on {device}...")
	model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
	model.to(device)

	# ### NEW: Dynamic Hardware Settings ###
	# If we have the 3090, we crank up the settings. If we are on a laptop, we keep them low.
	if device.type == "cuda":
		my_batch_size = 128        # 3090 has huge VRAM, we can process 64+ items at once
		my_fp16 = True            # "Mixed Precision" - 2x faster, uses half the VRAM
		my_workers = 4            # Use CPU cores to feed data to GPU faster
		print("GPU Mode: Batch Size 64, FP16 Enabled")
	else:
		my_batch_size = 16        # Safe for CPU/Laptops
		my_fp16 = False           # CPU cannot use FP16 effectively
		my_workers = 0            # Windows CPU sometimes struggles with workers > 0
		print("CPU Mode: Batch Size 16, FP16 Disabled")

	# Training configuration
	training_args = TrainingArguments(
		output_dir="./results",
		num_train_epochs=30,              # We set 10, but EarlyStopping will stop it sooner
		per_device_train_batch_size=my_batch_size, # ### NEW: Uses dynamic variable
		per_device_eval_batch_size=my_batch_size,  # ### NEW: Uses dynamic variable
		save_total_limit=2,
		warmup_steps=500,
		weight_decay=0.01,
		logging_dir="./logs",
		logging_steps=50,                 # Log more often since it's faster
		eval_strategy="epoch",
		save_strategy="epoch",
		load_best_model_at_end=True,
		fp16=my_fp16,                     # ### NEW: Massive speedup for 3090
		dataloader_num_workers=my_workers # ### NEW: Feeds data faster
	)

	# Create trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_tokenized,
		eval_dataset=test_tokenized,
		# ### NEW: Stop training early if accuracy doesn't improve for 3 epochs
		callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] 
	)

	# Train the model
	print("Starting training...")
	trainer.train()
	print("Training complete!")

	# Save the trained model
	print("Saving model to", model_save_path)
	model.save_pretrained(model_save_path)
	tokenizer.save_pretrained(model_save_path)
	
	return model

# Run the setup
model = setupModel()

# %% [markdown]
# For refrence, that took 17 minutes on my 3090. The results are:
# 
# Initializing model on cuda...
# 
# Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
# 
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# 
# GPU Mode: Batch Size 64, FP16 Enabled
# Starting training...
# 
# <div>
# 	<progress value='1176' max='5880' style='width:300px; height:20px; vertical-align: middle;'></progress>
# 	[1176/5880 16:35 < 1:06:29, 1.18 it/s, Epoch 6/30]
# </div>
# <table border="1" class="dataframe">
# <thead>
# <tr style="text-align: left;">
# 	<th>Epoch</th>
# 	<th>Training Loss</th>
# 	<th>Validation Loss</th>
# </tr>
# </thead>
# <tbody>
# <tr>
# 	<td>1</td>
# 	<td>0.279700</td>
# 	<td>0.229104</td>
# </tr>
# <tr>
# 	<td>2</td>
# 	<td>0.214900</td>
# 	<td>0.242618</td>
# </tr>
# <tr>
# 	<td>3</td>
# 	<td>0.148000</td>
# 	<td>0.194874</td>
# </tr>
# <tr>
# 	<td>4</td>
# 	<td>0.094900</td>
# 	<td>0.216039</td>
# </tr>
# <tr>
# 	<td>5</td>
# 	<td>0.062700</td>
# 	<td>0.264757</td>
# </tr>
# <tr>
# 	<td>6</td>
# 	<td>0.043600</td>
# 	<td>0.422895</td>
# </tr>
# </tbody>
# </table><p>
# Training complete!
# Saving model to ./saved_distilbert_model

# %%
# =============================================================================
# CELL 14: Evaluate BERT model
# Section: Comparative Classification (Section 7) - Embedding Model
# What: Getting predictions and calculating metrics for BERT
# =============================================================================


trainer = Trainer(
	model=model,
	args=TrainingArguments(output_dir="./results", per_device_eval_batch_size=16),
)

print("Getting predictions...")
bert_predictions_output = trainer.predict(test_tokenized)
bert_predictions = np.argmax(bert_predictions_output.predictions, axis=1)

# Calculate metrics
bert_accuracy = accuracy_score(test_subset["label"], bert_predictions)
bert_precision = precision_score(test_subset["label"], bert_predictions)
bert_recall = recall_score(test_subset["label"], bert_predictions)
bert_f1 = f1_score(test_subset["label"], bert_predictions)

print("\nEMBEDDING MODEL (DistilBERT)")
print("Accuracy: ", round(bert_accuracy, 4))
print("Precision:", round(bert_precision, 4))
print("Recall:   ", round(bert_recall, 4))
print("F1-Score: ", round(bert_f1, 4))

print("\nDetailed Classification Report:")
print(classification_report(test_subset["label"], bert_predictions, target_names=["Negative", "Positive"]))

# Save confusion matrix
bert_cm = confusion_matrix(test_subset["label"], bert_predictions)

# %%
# =============================================================================
# CELL 15: Compare all models
# Section: Performance Analysis (Section 9 in PDF)
# What: Creating summary table of all model results
# =============================================================================

results_df = pd.DataFrame({
	"Model": ["Baseline", "Naive Bayes", "DistilBERT"],
	"Accuracy": [baseline_accuracy, nb_accuracy, bert_accuracy],
	"Precision": [baseline_precision, nb_precision, bert_precision],
	"Recall": [baseline_recall, nb_recall, bert_recall],
	"F1-Score": [baseline_f1, nb_f1, bert_f1]
})

print("\nFINAL RESULTS COMPARISON")
print(results_df.to_string(index=False))

# %%
# =============================================================================
# CELL 16: Create visualizations
# Section: Performance Analysis (Section 9 in PDF)
# What: Creating comparison charts and confusion matrices
#       This generates publication-quality figures for your report
# =============================================================================

# Create a figure with 4 subplots arranged in 2 rows and 2 columns
# figsize=(15, 12) means width=15 inches, height=12 inches
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ============================================================================
# CHART 1: Bar chart comparing all metrics across all models
# Located at position [0, 0] (top-left)
# ============================================================================
ax1 = axes[0, 0]

# Create x-axis positions for each model (0, 1, 2 for Baseline, NB, BERT)
x = np.arange(len(results_df))

# Width of each bar (0.2 so we can fit 4 bars side by side)
width = 0.2

# Create 4 sets of bars, one for each metric
# x - width*1.5: Position first bar (Accuracy) to the left
# x - width*0.5: Position second bar (Precision) slightly left of center
# x + width*0.5: Position third bar (Recall) slightly right of center
# x + width*1.5: Position fourth bar (F1) to the right
ax1.bar(x - width*1.5, results_df["Accuracy"], width, label="Accuracy", alpha=0.8)
ax1.bar(x - width*0.5, results_df["Precision"], width, label="Precision", alpha=0.8)
ax1.bar(x + width*0.5, results_df["Recall"], width, label="Recall", alpha=0.8)
ax1.bar(x + width*1.5, results_df["F1-Score"], width, label="F1-Score", alpha=0.8)

# Set labels for x and y axes
ax1.set_xlabel("Model", fontsize=12)
ax1.set_ylabel("Score", fontsize=12)

# Set title for this chart
ax1.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")

# Set x-axis tick positions and labels
ax1.set_xticks(x)
ax1.set_xticklabels(results_df["Model"])

# Add legend to show which color represents which metric
ax1.legend()

# Set y-axis limits from 0 to 1 (since all metrics are between 0 and 1)
ax1.set_ylim([0, 1])

# Add horizontal grid lines for easier reading (alpha=0.3 makes them faint)
ax1.grid(axis="y", alpha=0.3)

# ============================================================================
# CHART 2: Confusion matrix heatmap for Naive Bayes
# Located at position [0, 1] (top-right)
# ============================================================================
ax2 = axes[0, 1]

# Create heatmap using seaborn
# nb_cm: confusion matrix data (2x2 array)
# annot=True: Show numbers in each cell
# fmt="d": Format numbers as integers
# cmap="Blues": Use blue color scheme
# xticklabels/yticklabels: Label the axes
sns.heatmap(nb_cm, annot=True, fmt="d", cmap="Blues", ax=ax2, 
			xticklabels=["Negative", "Positive"], 
			yticklabels=["Negative", "Positive"])

# Set title and axis labels
ax2.set_title("Confusion Matrix - Naive Bayes", fontsize=14, fontweight="bold")
ax2.set_ylabel("True Label", fontsize=12)
ax2.set_xlabel("Predicted Label", fontsize=12)

# ============================================================================
# CHART 3: Confusion matrix heatmap for DistilBERT
# Located at position [1, 0] (bottom-left)
# ============================================================================
ax3 = axes[1, 0]

# Similar to Chart 2 but with green color scheme and BERT data
sns.heatmap(bert_cm, annot=True, fmt="d", cmap="Greens", ax=ax3,
			xticklabels=["Negative", "Positive"], 
			yticklabels=["Negative", "Positive"])

ax3.set_title("Confusion Matrix - DistilBERT", fontsize=14, fontweight="bold")
ax3.set_ylabel("True Label", fontsize=12)
ax3.set_xlabel("Predicted Label", fontsize=12)

# ============================================================================
# CHART 4: Simple bar chart showing F1-Score for each model
# Located at position [1, 1] (bottom-right)
# ============================================================================
ax4 = axes[1, 1]

# Define colors for each bar (gray, blue, green)
colors = ["#808080", "#3498db", "#2ecc71"]

# Create bar chart with F1-scores
# alpha=0.8: Make bars slightly transparent
bars = ax4.bar(results_df["Model"], results_df["F1-Score"], color=colors, alpha=0.8)

# Set axis labels and title
ax4.set_ylabel("F1-Score", fontsize=12)
ax4.set_title("F1-Score Comparison", fontsize=14, fontweight="bold")
ax4.set_ylim([0, 1])

# Add grid lines for y-axis
ax4.grid(axis="y", alpha=0.3)

# Add text labels on top of each bar showing exact F1-score value
for bar in bars:
	# Get the height of this bar (which is the F1-score)
	height = bar.get_height()
	
	# Place text above the bar
	# bar.get_x() + bar.get_width()/2: Center horizontally on the bar
	# height: Place at the top of the bar
	# ha="center": Horizontal alignment center
	# va="bottom": Vertical alignment bottom (so text sits above bar)
	ax4.text(bar.get_x() + bar.get_width()/2., height,
			round(height, 4),
			ha="center", va="bottom", fontsize=10)

# Adjust spacing between subplots so they don't overlap
plt.tight_layout()

# Save the entire figure as a high-resolution PNG file
# dpi=300: High resolution for print quality
# bbox_inches="tight": Remove extra whitespace around figure
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")

# Display the figure
plt.show()

print("\nVisualization saved as 'model_comparison.png'")
print("Include this figure in your report!")

# %%



