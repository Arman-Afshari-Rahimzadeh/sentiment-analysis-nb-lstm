# 📊 Sentiment Analysis: Naïve Bayes vs LSTM  

## 📌 Project Overview  
Using the **IMDB dataset**, this project evaluates the performance of a traditional probabilistic model (**Naïve Bayes**) against a deep learning-based approach (**LSTM**) for **sentiment classification**. The goal is to compare their effectiveness and understand their strengths and limitations.  

## 🎯 Objectives  
✔ Implement **Naïve Bayes & LSTM models** for sentiment classification.  
✔ Compare performance using metrics like **accuracy, precision, recall, F1-score**.  
✔ Highlight strengths & weaknesses of each model for **real-world applications**.  
✔ Provide insights on when to use **statistical vs deep learning** approaches in NLP.  

## 📂 Dataset Information  
Due to size limitations, the dataset is not stored in this repository.  
📥 **Download the IMDB Dataset here:** [Click Here](https://drive.google.com/file/d/1Zf-8WZFhGql2Qy1agjHtDS1owm-jS9Zd/view?usp=sharing)  

## 🛠️ Technologies & Tools Used  
- **Programming Language**: Python  
- **Libraries**: NumPy, Pandas, Scikit-learn, TensorFlow, Keras, NLTK  
- **Feature Extraction**: TF-IDF, Word Embeddings  
- **Modeling**: Naïve Bayes (MultinomialNB), LSTM  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score  

## 🏗️ Project Structure  
sentiment-analysis-nb-lstm
│── IMDB Dataset.csv # Dataset (not included, download separately)
│── sentiment_analysis.ipynb # Jupyter Notebook with code
│── report.pdf # Research report with methodology & results
│── README.md # Project documentation

bash
Copy
Edit

## 📊 Results & Insights  
✔ **Naïve Bayes**: Faster & computationally efficient, but lacks deep contextual understanding.  
✔ **LSTM**: More accurate, captures complex relationships, but requires more data & processing power.  
✔ **Conclusion**: Use NB for quick insights; use LSTM when deep semantic understanding is required.  

## 🚀 How to Run the Project  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/Arman-Afshari-Rahimzadeh/sentiment-analysis-nb-lstm.git
cd sentiment-analysis-nb-lstm
2️⃣ Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run the Jupyter Notebook

bash
Copy
Edit
jupyter notebook
4️⃣ Open sentiment_analysis.ipynb and execute the cells.

📝 Future Improvements
🔹 Try additional models (BERT, Transformer-based models).
🔹 Optimize hyperparameters for better accuracy.
🔹 Expand dataset for more generalization.

📬 Contact & Connect
📧 Email: proarman1@gmail.com
🔗 GitHub: Arman-Afshari-Rahimzadeh
🔗 LinkedIn: Arman Afshari-Rahimzadeh
