from preprocess import TextPreprocessor
from train import ModelTrainer

load = TextPreprocessor.load_and_preprocess_text('C:/Users/WilmarAl/Documents/00-Personal/01 - Maestria/00 - UNIVERSIDAD CENTRAL/4 Semestre/Deep Learning/Repos/generador_texto/texto.txt')

model_trainer = ModelTrainer()
model_trainer.load_model()
text_preprocessor = TextPreprocessor()
text_preprocessor.load_preprocessing_data()