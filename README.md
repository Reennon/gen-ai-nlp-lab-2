# gen-ai-nlp-lab-2

### Overview
This project uses the MATH-500 dataset to build a math problem-solving pipeline in Ukrainian. The workflow includes dataset translation, cleaning, embedding generation, and example retrieval using FAISS. We implemented RAG-based generation with minimal preprocessing and evaluated the results in Jupyter Notebooks.

---

### Workflow Description
1. **Dataset**: Use [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) and translate it to Ukrainian with DeepL (`notebooks/preprocessing.ipynb`).
2. **Cleaning**: Remove extra texts like visualization code.
3. **Embedding Text**: Concatenate fields:
   - `problem`: Problem description.
   - `solution`: Solution explanation.
   - `subject`: Topic (e.g., geometry, algebra).
   - `level`: Difficulty level.
4. **Vectorization**: Generate embeddings with OpenAI's `ada-002` model.
5. **Storage**: Save results to CSV and FAISS `.index` files.

---

### Screenshots

![Screenshot 2025-01-08 at 22.18.09.png](./images/Screenshot%202025-01-08%20at%2022.18.09.png)
![Screenshot 2025-01-08 at 22.18.16.png](./images/Screenshot%202025-01-08%20at%2022.18.16.png)
![Screenshot 2025-01-08 at 22.18.28.png](./images/Screenshot%202025-01-08%20at%2022.18.28.png)
![Screenshot 2025-01-08 at 22.18.33.png](./images/Screenshot%202025-01-08%20at%2022.18.33.png)
![Screenshot 2025-01-08 at 22.18.46.png](./images/Screenshot%202025-01-08%20at%2022.18.46.png)
![Screenshot 2025-01-08 at 22.18.52.png](./images/Screenshot%202025-01-08%20at%2022.18.52.png)
![Screenshot 2025-01-08 at 22.18.58.png](./images/Screenshot%202025-01-08%20at%2022.18.58.png)
![Screenshot 2025-01-08 at 22.19.28.png](./images/Screenshot%202025-01-08%20at%2022.19.28.png)
![Screenshot 2025-01-08 at 22.19.46.png](./images/Screenshot%202025-01-08%20at%2022.19.46.png)

### Points to improve

- Improve evaluation, as we can see most of the time, the answer is correct, but the solution might be different from the target, which is expected, as in math often it is that there's no SINGLE but MANY possible solutions; So we got to either count more on whether we got the answer right, or whether the solution is plausible or at least partially correct.
- Use more mature frameworks: LangFuse, LangChain, LangGraph, LanceDB, Streamlit; Due to limited time, the solution implemented in Jupyter Notebooks with minimal MLOps frameworks.
- Experiment with more prompts, and implement two-stage prompting, with one model selecting appropriate similar examples, retrieved by cosine similarity, but with LLM, and cleaning up the data; and other doing the answering.
- Improve evaluation to have a fallback and GEval or other frameworks. (Btw, DeepEval collect all data to their servers, so, maybe there are alternative approaches, that does not monitor your data and more ethical!)


### Notebooks Structure
- `notebooks/evaluation.ipynb` Evaluation based on test set (10 samples) of translated MATH-500 dataset
- `notebooks/preprocessing.ipynb` Preprocesses the MATH-500 dataset to translate, cleanup, split and vectorize it
- `notebooks/processing.ipynb` Actual processing and demo of generating examples using our agent rag approach

### Datasets Structure

The files are not present in GitHub, however, you can download them using [Google Drive Link](https://drive.google.com/drive/folders/1DL-MdQCl29coYUzADGUDpTNnIRmZn0LW?usp=sharing)

- `datasets/math-500-uk.csv` Translated with DeepL dataset
- `datasets/math-500-uk.index` Translated with DeepL FAISS vector database
- `datasets/math-500-uk-inference.csv` Inference (i.e. train) part of our translated dataset, used for similarity search (490 examples)
- `datasets/math-500-uk-test.csv` Test part of our translated dataset, used for evaluation (10 examples)
- `datasets/math-500-uk-with-embeddings.csv` Temporary CSV file with embeddings from OpenAI for a fallback on vectorization

### Conclusions

The system generates accurate and fluent solutions in Ukrainian. Minimal cleaning caused some issues, such as untranslated visualization blocks and over-translated LaTeX commands (e.g., \triangle to \трикутник). However, due to a concise and additional cleanup instructions the issue was alleviated and didn't cause any additional errors. The RAG-agent pipeline performed well despite these challenges, and OpenAI's gpt-4o-mini handled Ukrainian fluently, making it a solid choice for such tasks. The project successfully demonstrated the feasibility of math problem-solving in a low-resource language.

