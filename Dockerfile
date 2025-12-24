FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY MLProject/conda.yaml .

# Install dependencies
RUN pip install pandas numpy scikit-learn==1.5.2 category_encoders mlflow dagshub optuna matplotlib seaborn

# Copy model files
COPY MLProject/ .

# Run training
CMD ["python", "modelling.py"]
