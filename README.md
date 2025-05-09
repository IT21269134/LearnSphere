# 📚 LearnSphere: Conversational PDF Question Answering App

Welcome to **LearnSphere** — a Streamlit-based app that empowers users to interact with PDF documents using natural language queries. Perfect for students, educators, and researchers who want detailed insights from lecture notes or research papers, fast.

---

## 🚀 Features

- Upload multiple PDF files
- Extract and chunk content intelligently
- Generate vector embeddings using Google Generative AI
- Ask context-aware questions about uploaded content
- Conversational responses powered by Gemini AI
- Simple, clean, and responsive Streamlit UI

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/learnsphere.git
cd learnsphere
```
### 2. Create a Virtual Environment

```bash
python -m venv VM_environment_name
```
### 3. Activate the Virtual Environment
```bash
VM_environment_name\Scripts\activate.bat  # For Windows
(For Unix/macOS, use: source VM_environment_name/bin/activate)
```

### 4. Install Dependencies
Make sure you have a .env file in the root directory with your GOOGLE_API_KEY. Then run:

```bash
pip install -r requirements.txt
```

### 5. Run the Application
Navigate to the project folder and launch the Streamlit app:

```bash
streamlit run app.py
```


