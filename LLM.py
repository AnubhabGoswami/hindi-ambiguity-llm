

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings, random, os

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()   # mixed precision only on GPU

print(f"PyTorch  : {torch.__version__}")
print(f"CUDA     : {torch.cuda.is_available()}")
print(f"AMP      : {USE_AMP}")
print(f"Device   : {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
#  LABEL 0 — Lexical ambiguity  (word has multiple meanings)
# ──────────────────────────────────────────────────────────────────────────────
LEX = [
    # आम  (mango / ordinary)
    ("मुझे आम खाना है।", 0),
    ("यह एक आम बात है।", 0),
    ("बाजार में आम मिल रहे हैं।", 0),
    ("यह आम इंसान नहीं है।", 0),
    ("वह आम की तरह मीठा है।", 0),
    # काला  (black / art)
    ("यह काला रंग है।", 0),
    ("उसे काला सीखना है।", 0),
    ("उसके काले बाल हैं।", 0),
    ("काला जादू खतरनाक होता है।", 0),
    # पत्ता  (leaf / card / clue)
    ("पेड़ से पत्ता गिरा।", 0),
    ("उसने पत्ता फेंक दिया।", 0),
    ("उसके हाथ में पत्ता था।", 0),
    ("पत्ता खेलना उसे पसंद है।", 0),
    # सर  (head / lake / sir)
    ("उसका सर दर्द कर रहा है।", 0),
    ("हम सर के पास गए।", 0),
    ("सर ने क्लास ली।", 0),
    # कान  (ear / crow-dialectal)
    ("उसके कान बड़े हैं।", 0),
    ("कान पेड़ पर बैठा है।", 0),
    ("कान से सुनो।", 0),
    # काल  (time / death / tomorrow-dialectal)
    ("वह काल के गाल में समा गया।", 0),
    ("काल मैं दिल्ली जाऊंगा।", 0),
    ("काल बड़ा कठिन है।", 0),
    # बैल  (ox / bail)
    ("बैल खेत में काम कर रहा है।", 0),
    ("उसे जेल से बेल मिल गई।", 0),
    # डंडा  (stick / officer-slang)
    ("बच्चे ने डंडा उठाया।", 0),
    ("डंडा आ गया, सब भाग गए।", 0),
    # फल  (fruit / result)
    ("यह पेड़ अच्छे फल देता है।", 0),
    ("मेहनत का फल मीठा होता है।", 0),
    ("परीक्षा का फल आ गया।", 0),
    # नेता  (leader / vein-archaic)
    ("वह एक अच्छा नेता है।", 0),
    ("देश का नेता जिम्मेदार होता है।", 0),
    # पानी  (water / weak)
    ("मुझे पानी पीना है।", 0),
    ("वह तो बिल्कुल पानी निकला।", 0),
    # मन  (mind / unit of weight)
    ("उसका मन उदास है।", 0),
    ("एक मन अनाज तौलो।", 0),
    # खाल  (skin / leather)
    ("बाघ की खाल मूल्यवान है।", 0),
    ("उसने खाल उतार ली।", 0),
    # ताल  (pond / rhythm)
    ("ताल में मछलियाँ हैं।", 0),
    ("संगीत में ताल बहुत जरूरी है।", 0),
    # कटना  (to be cut / to pass time)
    ("रस्सी कट गई।", 0),
    ("समय कट रहा है।", 0),
    ("दिन कट गया।", 0),
]

# ──────────────────────────────────────────────────────────────────────────────
#  LABEL 1 — Syntactic / Structural ambiguity
# ──────────────────────────────────────────────────────────────────────────────
SYN = [
    # PP / instrument attachment
    ("उसने लाठी लेकर आदमी को देखा।", 1),
    ("उसने चाकू से आदमी को मारा।", 1),
    ("उसने फूल लेकर लड़की को दिया।", 1),
    ("उसने छड़ी से कुत्ते को भगाया।", 1),
    ("राम ने किताब लेकर मोहन को बुलाया।", 1),
    # Relative clause attachment
    ("वह आदमी जो मेरे घर आया थका हुआ था।", 1),
    ("मैं उसे बाज़ार में मिली लड़की से मिला।", 1),
    ("उसने वह किताब खरीदी जो महँगी थी।", 1),
    ("वह लड़का जो दौड़ता था स्कूल गया।", 1),
    ("मैंने उस आदमी को देखा जो वहाँ था।", 1),
    # Coordination ambiguity
    ("लड़की और लड़के ने दौड़ लगाई जो थके हुए थे।", 1),
    ("बच्चे और माँ ने मिलकर खाना बनाया।", 1),
    ("राम और सीता और लक्ष्मण वन गए।", 1),
    ("उसने सेब और आम खाए जो मीठे थे।", 1),
    # Nested subject / coreference
    ("राम ने श्याम को मारा जब वह आया।", 1),
    ("उसने कहा कि वह जाएगा।", 1),
    ("उसने बताया कि कल वह आएगा।", 1),
    ("मोहन ने सोहन से कहा कि वह गलत है।", 1),
    ("उसने उससे पूछा कि वह कहाँ जाएगा।", 1),
    # Participle / adjunct attachment
    ("राम ने सेब खाते हुए गाना सुना।", 1),
    ("वह दौड़ता हुआ लड़का स्कूल गया।", 1),
    ("छत पर बैठे आदमी ने गाना गाया।", 1),
    ("उसने सोते हुए बच्चे को उठाया।", 1),
    ("वह हँसते हुए कमरे में आई।", 1),
    # Number agreement ambiguity
    ("मेरा दोस्त और उसका भाई आएगा।", 1),
    ("राम या श्याम आएगा।", 1),
    ("लड़का या लड़कियाँ जाएंगी।", 1),
    # Subject of action ambiguity
    ("वह लड़की को देखकर मुस्कुराया।", 1),
    ("मैंने उसे मारते देखा।", 1),
    ("उसने उसे रोते देखा।", 1),
    ("वह उसे जाते हुए मिली।", 1),
    # Other structural
    ("कल से काम शुरू होगा।", 1),
    ("उसने घर जाकर खाना खाया।", 1),
    ("वह आ रहा था तभी बारिश हुई।", 1),
    ("मैंने उसे देखा जो रो रहा था।", 1),
    ("वह जल्दी-जल्दी घर पहुँचा।", 1),
    ("राम ने कहा वह पढ़ेगा।", 1),
    ("उसने बच्चे को सोते समय देखा।", 1),
    ("वह आया और चला गया।", 1),
    ("मैं उससे मिला जो बाहर था।", 1),
    ("बच्चा खेलते-खेलते सो गया।", 1),
    ("उसने उसे बुलाया जो पास था।", 1),
    ("राम ने वह किताब रखी जो पुरानी थी।", 1),
    ("वह लड़की नाचते हुए गिर गई।", 1),
]

# ──────────────────────────────────────────────────────────────────────────────
#  LABEL 2 — Semantic / Scope ambiguity
# ──────────────────────────────────────────────────────────────────────────────
SEM = [
    # Negation scope
    ("वह हमेशा झूठ नहीं बोलता।", 2),
    ("मैं उसे नहीं जानता शायद।", 2),
    ("उसने शायद नहीं जाना चाहा।", 2),
    ("वह कभी-कभी नहीं आता है।", 2),
    ("वह जल्दी-जल्दी कभी नहीं बोलता।", 2),
    ("मुझे नहीं पता कि वह आएगा।", 2),
    ("यह काम आसान नहीं है शायद।", 2),
    ("वह हमेशा सच नहीं बोलता।", 2),
    ("वह कभी नहीं आता शायद।", 2),
    # Quantifier scope (universal / existential)
    ("हर छात्र ने कोई किताब नहीं पढ़ी।", 2),
    ("कुछ लोग सभी जानवरों से प्यार करते हैं।", 2),
    ("सभी बच्चे कोई खेल जानते हैं।", 2),
    ("हर आदमी एक औरत से प्यार करता है।", 2),
    ("कम से कम दो छात्र पास नहीं हुए।", 2),
    ("उसे सभी ने नहीं देखा।", 2),
    ("हर लड़की कोई गाना जानती है।", 2),
    ("कुछ छात्र सभी परीक्षाएँ पास करते हैं।", 2),
    ("हर दिन कुछ न कुछ होता है।", 2),
    ("सभी ने कुछ न कुछ खाया।", 2),
    # Modal + epistemic scope
    ("मुझे लगता है कि शायद वह आए।", 2),
    ("वह जरूर कहीं गया होगा।", 2),
    ("शायद वह कल आएगा।", 2),
    ("लगता है वह थका हुआ है।", 2),
    ("वह आ सकता है या नहीं।", 2),
    # Disjunction scope
    ("राम या श्याम में से कोई एक आएगा।", 2),
    ("वह न तो खाता है न सोता है।", 2),
    ("या तो वह जाएगा या मैं जाऊंगा।", 2),
    ("वह भी आएगा और तुम भी।", 2),
    # Frequency + negation
    ("मैं हर दिन कुछ न कुछ सीखता हूँ।", 2),
    ("वह रोज़ कभी-कभी देर से आता है।", 2),
    ("वह कभी-कभी देर से सोता है।", 2),
    # Belief + scope
    ("उसे विश्वास है कि वह जीतेगा।", 2),
    ("मुझे लगता है सब ठीक हो जाएगा।", 2),
    ("उसे यकीन नहीं कि वह सही है।", 2),
    # Other scope ambiguities
    ("दो से अधिक बच्चे नहीं आए।", 2),
    ("वह अक्सर नहीं सोता।", 2),
    ("सब कुछ ठीक नहीं है।", 2),
    ("हर बार ऐसा नहीं होता।", 2),
    ("कोई नहीं आया शायद।", 2),
    ("वह शायद कभी नहीं बदलेगा।", 2),
    ("कम लोगों ने ज्यादा काम किया।", 2),
    ("उसने सबको कुछ न कुछ दिया।", 2),
    ("हर बात पर वह सहमत नहीं है।", 2),
    ("शायद कोई आएगा।", 2),
]

ALL_DATA = LEX + SYN + SEM
df = pd.DataFrame(ALL_DATA, columns=["sentence", "label"])
label_names = {0: "Lexical", 1: "Syntactic", 2: "Semantic"}
df["label_name"] = df["label"].map(label_names)

print(f"Total samples : {len(df)}")
print(df["label_name"].value_counts().to_string())
print("\nSample rows:")
print(df.head(6).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Hindi NLP Ambiguity Dataset — expanded", fontsize=14, fontweight="bold")
counts = df["label_name"].value_counts()
colors = ["#1D9E75", "#7F77DD", "#EF9F27"]

axes[0].bar(counts.index, counts.values, color=colors, edgecolor="black", width=0.5)
axes[0].set_title("Class distribution")
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140)
axes[1].set_title("Proportion")

plt.tight_layout()
plt.savefig("class_distribution.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Distribution chart saved.")

# Hold out 15 % as the final test set — never touched during CV
train_full_df, test_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)
print(f"Train+CV pool : {len(train_full_df)}  |  Held-out test : {len(test_df)}")
print("Test class balance:")
print(test_df["label_name"].value_counts().to_string())

# ── Configuration ────────────────────────────────────────────────────────────
# Switch from giant 110M-param MuRIL fine-tuning to:
#   sentence-transformers MiniLM (118 MB) → fixed embeddings → sklearn SVM
# This is the correct approach for ~110 labelled samples.
ENCODER_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
N_FOLDS      = 5
SEED         = 42
NUM_LABELS   = 3
label_names  = {0: "Lexical", 1: "Syntactic", 2: "Semantic"}

print(f"Encoder : {ENCODER_NAME}")
print(f"CV folds: {N_FOLDS}")
print(f"Samples : {len(df)}")

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer(ENCODER_NAME)
print(f"Encoder loaded — embedding dim: {encoder.get_sentence_embedding_dimension()}")

# Encode ALL sentences once (fast, ~2 s on CPU)
sentences = df["sentence"].tolist()
labels    = df["label"].values

X = encoder.encode(sentences, batch_size=32, show_progress_bar=True,
                   normalize_embeddings=True)  # cosine-friendly unit vectors
print(f"Embedding matrix shape: {X.shape}")

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate

# SVM with RBF kernel — best default for fixed embeddings at this data scale
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=5.0, gamma="scale",
                   probability=True, random_state=SEED)),
])

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# Use the same train_full_df split established in Cell 3
train_idx    = train_full_df.index.tolist()
X_train_full = X[train_idx]
y_train_full = labels[train_idx]

cv_results = cross_validate(
    clf_pipeline, X_train_full, y_train_full,
    cv=skf, scoring="accuracy", return_train_score=True,
)

fold_accs = cv_results["test_score"]
print(f"Running {N_FOLDS}-Fold CV on {len(X_train_full)} samples...\n")
print(f"  {'Fold':>4}  {'Val Acc':>9}")
print("  " + "─" * 16)
for i, acc in enumerate(fold_accs, 1):
    print(f"  {i:>4}  {acc:>9.4f}")
print("  " + "─" * 16)
print(f"  Mean  {fold_accs.mean():>9.4f}  ± {fold_accs.std():.4f}")
print(f"\n CV complete. Mean val accuracy: {fold_accs.mean():.4f}")

# ── Final model: fit on ALL train_full data ───────────────────────────────────
final_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm",    SVC(kernel="rbf", C=5.0, gamma="scale",
                   probability=True, random_state=SEED)),
])
final_clf.fit(X_train_full, y_train_full)
print(" Final classifier fitted on full training pool.")

# ── Held-out test evaluation ─────────────────────────────────────────────────
test_idx = test_df.index.tolist()
X_test   = X[test_idx]
y_test   = labels[test_idx]

test_preds = final_clf.predict(X_test)
test_acc   = accuracy_score(y_test, test_preds)

print(f"Test Accuracy : {test_acc:.4f}\n")
print("Classification Report:")
print(classification_report(
    y_test, test_preds,
    target_names=list(label_names.values()),
    zero_division=0,
))
print(f"5-Fold CV Mean ± Std  : {fold_accs.mean():.4f} ± {fold_accs.std():.4f}")

# ── Confusion matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(label_names.values()),
            yticklabels=list(label_names.values()))
plt.title("Confusion matrix — held-out test set", fontsize=13, fontweight="bold")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print(" Confusion matrix saved.")

# ── CV + split accuracy chart (replaces training-curves plot) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SVM + MiniLM — Cross-validation results", fontsize=13, fontweight="bold")

fold_labels = [f"Fold {i}" for i in range(1, N_FOLDS + 1)]
colors_cv   = ["#1D9E75", "#7F77DD", "#EF9F27", "#E05C5C", "#4DA6E8"]

axes[0].bar(fold_labels, fold_accs, color=colors_cv, edgecolor="black", width=0.5)
axes[0].axhline(fold_accs.mean(), color="red", linestyle="--",
                label=f"Mean={fold_accs.mean():.3f}")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("Per-fold validation accuracy")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
for i, v in enumerate(fold_accs):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

train_acc_full = accuracy_score(y_train_full, final_clf.predict(X_train_full))
split_labels   = ["Train (full)", "CV Mean", "Test"]
split_values   = [train_acc_full, fold_accs.mean(), test_acc]
bar_colors     = ["#7F77DD", "#1D9E75", "#EF9F27"]
axes[1].bar(split_labels, split_values, color=bar_colors, edgecolor="black", width=0.5)
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Train / CV / Test accuracy")
axes[1].set_ylabel("Accuracy")
for i, v in enumerate(split_values):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("cv_accuracy.png", dpi=120, bbox_inches="tight")
plt.show()
print(" CV accuracy chart saved.")

# ── 2-D embedding visualisation (UMAP or t-SNE fallback) ────────────────────
# Shows whether the three ambiguity classes form separable clusters
try:
    from umap import UMAP
    reducer      = UMAP(n_components=2, random_state=SEED)
    reducer_name = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer      = TSNE(n_components=2, random_state=SEED, perplexity=20)
    reducer_name = "t-SNE"

X_2d       = reducer.fit_transform(X)
colors_map = {0: "#1D9E75", 1: "#7F77DD", 2: "#EF9F27"}

plt.figure(figsize=(8, 6))
for lbl, name in label_names.items():
    mask = labels == lbl
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors_map[lbl], label=name,
                alpha=0.75, edgecolors="k", linewidths=0.3)
plt.title(f"{reducer_name} of MiniLM embeddings — all {len(df)} sentences", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("embedding_viz.png", dpi=120, bbox_inches="tight")
plt.show()
print(f" {reducer_name} visualisation saved.")

# ── Interactive classifier ────────────────────────────────────────────────────
def predict_one(sentence):
    """Encode sentence and return (pred_label_id, prob_array)."""
    emb   = encoder.encode([sentence], normalize_embeddings=True)
    probs = final_clf.predict_proba(emb)[0]   # shape (3,)
    pred  = int(np.argmax(probs))
    return pred, probs


def confidence_bar(prob, width=20):
    filled = int(round(prob * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {prob*100:5.1f}%"


print("=" * 56)
print("  Hindi Ambiguity Classifier — interactive mode")
print("  Model : MiniLM encoder + SVM (RBF kernel)")
print("  Type a Hindi sentence.  Enter 'quit' to exit.")
print("=" * 56)

while True:
    sentence = input("\nSentence: ").strip()
    if sentence.lower() in ("quit", "exit", "q", ""):
        print("Goodbye!")
        break
    pred_id, probs = predict_one(sentence)
    print(f"\nPrediction : {label_names[pred_id].upper()}")
    print()
    for i, name in label_names.items():
        print(f"  {name:<12} {confidence_bar(probs[i])}")

