import os


import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import streamlit as st

# OpenAI API Key (Replace with your key)
openai.api_key = "your-api-key"

# Initialize LangChain LLM
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Updated Prompt Template with Deep Learning, Transfer Learning, and Visualization Techniques
from langchain.prompts import PromptTemplate

# Define the final comprehensive and impactful prompt for machine learning and deep learning models
prompt = PromptTemplate(
    input_variables=["problem_description", "dataset_description"],
    template="""
    Given the following **problem description**:
    {problem_description}

    And the dataset details:
    {dataset_description}

    Please provide the **best solution** (either Machine Learning, Deep Learning, or Generative AI) for solving this problem, but only recommend **one approach** (ML, DL, or Generative AI) based on the problem and dataset. If multiple approaches are viable, suggest the more **accurate and efficient** approach. Consider **pre-built models** and **vector database integration** where relevant.

    **1. Data Preprocessing**:
        - Suggest the **initial data preprocessing** steps for the dataset.
        - Common preprocessing steps include:
            - Handling **missing values** (e.g., imputation or removal).
            - **Feature scaling** (e.g., **StandardScaler**, **Min-Max scaling**) for ML and DL models.
            - **Encoding categorical variables** (e.g., **One-Hot Encoding**, **Label Encoding**).
            - For **image data**, suggest **image normalization**.
            - For **NLP tasks**, suggest **tokenization**, **stopword removal**, and **lemmatization**.

    **2. Feature Engineering**:
        - Recommend **feature engineering** techniques tailored to the dataset:
            - **For ML models**: Feature selection, interaction terms, polynomial features, etc.
            - **For DL models**: Data augmentation (for image and text data) and time-series transformations (e.g., lag features, rolling windows).
            - **For NLP tasks (Generative AI)**: Embeddings (e.g., **Word2Vec**, **GloVe**, **BERT embeddings**) or other vectorization methods.
            - **For time-series tasks**, recommend lag features, trend decomposition, and seasonality handling.

    **3. Algorithm Selection**:
        - Based on the problem type (e.g., regression, classification, time-series, NLP), identify the **best algorithm** (either **ML**, **DL**, or **Generative AI**).
        - If **Machine Learning** is selected:
            - Suggest algorithms like **SVM**, **Random Forest**, **XGBoost**, **Logistic Regression**, etc.
        - If **Deep Learning** is selected:
            - Suggest models like **CNN**, **LSTM**, **Transformer**, **GANs**, etc.
        - If **Generative AI** is selected (for NLP tasks):
            - Suggest pre-trained models like **GPT**, **BERT**, **T5**, **BART**, etc., or fine-tuned models for the specific task.
        - Ensure that only one model type (ML, DL, or Generative AI) is recommended based on dataset and problem type.

    **4. Model Configuration**:
        - Provide configuration recommendations for the chosen model type:
            - **For ML models**: 
                - Suggest **hyperparameters** such as **C**, **kernel**, **max_depth**, **learning_rate**, etc.
                - Indicate if **feature scaling** is needed for specific algorithms (e.g., SVM, KNN).
            - **For DL models**: 
                - Recommend architecture types (e.g., **Convolutional layers** for CNN, **LSTM layers** for RNNs).
                - Suggest **activation functions** (e.g., **ReLU**, **Tanh**, **Sigmoid**).
                - Recommend regularization methods (e.g., **dropout**, **batch normalization**).
            - **For Generative AI models**: 
                - Suggest the use of **pre-trained models** (e.g., **GPT-3/4**, **T5**, **BERT**, **LangChain**) and fine-tuning techniques for specific tasks.
                - For **semantic search**, recommend vector databases like **FAISS**, **ChromaDB**, or **Pinecone** for storing and retrieving vectors.
                - For NLP tasks, consider using **transformers**, **tokenizers**, and integration with **LangChain** for advanced use cases like question answering or document retrieval.

    **5. Hyperparameter Tuning**:
        - Recommend **tuning strategies** for the selected model type:
            - **For ML algorithms** (e.g., **SVM**, **Random Forest**):
                - Hyperparameters like **C**, **gamma**, **max_depth**, and **number of estimators** should be tuned.
                - Suggest tuning methods like **GridSearchCV** or **RandomizedSearchCV**.
            - **For DL models**:
                - Tune hyperparameters such as **learning rate**, **batch size**, **epochs**, **dropout rate**, and **optimizer** (e.g., **Adam**, **SGD**).
            - **For Generative AI models**:
                - Fine-tune learning rates, batch sizes, and **warm-up steps** for **transfer learning**.
                - Suggest evaluation methods for **text generation**, **question answering**, or **semantic search** using **metrics** like **BLEU score**, **ROUGE**, and **accuracy**.

    **6. Model Evaluation**:
        - Evaluate the performance using appropriate **metrics**:
            - **For ML models**: Use **accuracy**, **precision**, **recall**, **F1-score**, and **AUC** for classification tasks, or **RMSE**, **MAE** for regression tasks.
            - **For DL models**: Similar to ML, with additional metrics like **IoU** (for segmentation) or **BLEU** (for sequence generation).
            - **For Generative AI models**:
                - Evaluate **text generation** with metrics like **BLEU**, **ROUGE**, and **Perplexity**.
                - Evaluate **semantic search** using **mean reciprocal rank (MRR)** and **retrieval accuracy**.

    **7. Mitigating Overfitting/Underfitting**:
        - For **ML models**:
            - Use **cross-validation**, **regularization**, and tune hyperparameters like **C**, **gamma** for SVM.
        - For **DL models**:
            - Suggest **dropout**, **early stopping**, and **data augmentation** for overfitting prevention.
        - For **Generative AI models**:
            - Use **early stopping** and tune **learning rates**.
            - Consider **knowledge distillation** or **model pruning** for more efficient models.

    **8. Deployment and Roadmap**:
        - **For ML models**:
            - Suggest deployment tools like **Flask**, **FastAPI**, or **Streamlit** for creating REST APIs.
        - **For DL models**:
            - Recommend frameworks like **TensorFlow Serving**, **ONNX**, or **TensorFlow Lite** for deploying DL models.
        - **For Generative AI models**:
            - Use **HuggingFace** or **LangChain** to deploy and serve models.
            - For **semantic search**, use **FAISS**, **ChromaDB**, or **Pinecone** for fast retrieval.
            - For complex NLP tasks, suggest integrating with **LangChain** to enable **document retrieval**, **QA systems**, and **semantic search**.

    **9. Final Recommendations**:
        - Provide a comprehensive summary:
            - The **best approach** (ML, DL, or Generative AI) based on the problem and dataset.
            - The **recommended algorithm(s)**, **architecture**, and **hyperparameters**.
            - **Evaluation metrics** suitable for the task.
            - **Deployment strategy** and **scaling** techniques.
            - If applicable, recommend using **pre-trained models** (e.g., **GPT**, **BERT**) or **Generative AI frameworks** like **LangChain** for NLP tasks.
            - Suggest integrating **vector databases** (e.g., **FAISS**, **ChromaDB**) for **semantic search**, **document retrieval**, or **memory-based learning**.
    """
)

# Define LangChain LLMChain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Streamlit UI
st.title("AI-Powered ML/DL/Generative AI Solution ")

problem_description = st.text_area("Describe Your Problem:")
dataset_description = st.text_area("Describe Your Dataset:")

if st.button("Get Solution"):
    response = llm_chain.run({
        "problem_description": problem_description,
        "dataset_description": dataset_description
    })

    st.subheader("Recommended Solution:")
    st.write(response)
