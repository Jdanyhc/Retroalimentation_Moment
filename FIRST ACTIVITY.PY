
import torch # import torcch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def Sentiment_Analysis(): #Function for detect the positive or negative sentiment
  tokenizer = AutoTokenizer.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
  model = AutoModelForSequenceClassification.from_pretrained("Seethal/sentiment_analysis_generic_dataset")
  f = open("tiny_movie_reviews_dataset (1).txt", "r") #Uploaded the txt
  data = f.readlines() #Lecture of lines for txt
  for idx, line in enumerate(data, start=1): #enumerate respuest
    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=512) #define the lenght of the lines and tokenize 
    with torch.no_grad():
      logits = model(**inputs).logits # lecture the inputs
      predicted_class_id = logits.argmax().item() 
      if predicted_class_id == 1: #check the result of tests and classify
        print(str(idx) + ': POSITIVE')
      else:
        print(str(idx) + ': NEGATIVE')

Sentiment_Analysis() 
