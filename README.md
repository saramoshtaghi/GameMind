# 🎮 GameQA — Question Answering on Gaming Texts with BERT

GameQA is a fine-tuned BERT-based Question Answering system trained to understand and extract answers from gaming-related documents and texts. Whether you're asking about a game’s release date or a character's background, GameQA finds the right answer directly from the content.

![GameQA Logo](./assets/gameqa_logo.png)

## 📌 Project Goals

The goal is to build a domain-specific QA model that can:
- Understand questions about video games
- Retrieve short, accurate answers from related text
- Leverage the power of transformer-based models for extractive QA

## 🧠 Model

- Base model: `bert-base-uncased`
- Fine-tuned for Question Answering using Hugging Face Trainer
- Trained on a custom dataset of QA pairs from gaming contexts

## 📊 Dataset Creation & Fine-Tuning

This project was developed by **Sara Moshtaghi**, who:
- Collected gaming-related content from Wikipedia using the Wikipedia API  
- Applied sentence segmentation with `spaCy`  
- Generated QA pairs and converted them into **SQuAD format**  
- Fine-tuned the model using Hugging Face’s `Trainer` API  

The dataset and model are specifically crafted to perform well in the gaming domain.

## 🛠️ Tools & Libraries

- Python 🐍
- Hugging Face Transformers 🤗
- PyTorch
- scikit-learn
- spaCy
- Datasets

## 🧪 Running Inference

To ask questions dynamically from the fine-tuned QA model:

```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import pandas as pd

# Load fine-tuned model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("models/gameqa")
tokenizer = AutoTokenizer.from_pretrained("models/gameqa")
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load context from the dataset
df = pd.read_csv("data/gaming_qa_dataset.csv")
titles = df["title"].unique().tolist()

# Show available games
print("🎮 Choose a game:")
for idx, title in enumerate(titles):
    print(f"{idx+1}. {title}")
selected_index = int(input("Enter number: ")) - 1
selected_title = titles[selected_index]
context = df[df["title"] == selected_title]["context"].values[0]

# Ask question
question = input(f"Ask a question about {selected_title}: ")
result = qa_pipeline(question=question, context=context)
print(f"\n📣 Answer: {result['answer']}")
```

## 🗂️ Project Structure

```text
GameQA/
├── assets/                    # Logo and visuals
├── data/                      # QA dataset (CSV and SQuAD JSON)
├── models/                    # Fine-tuned model and tokenizer
├── notebooks/                # EDA and training walkthroughs
├── scripts/                  # Utility and training scripts
├── README.md
```

## 👤 Author

**Sara Moshtaghi** — NLP Researcher & Machine Learning Engineer  
🔗 [LinkedIn](https://linkedin.com/in/saramoshtaghi) | 🤗 [Hugging Face](https://huggingface.co/SaraWonder)
