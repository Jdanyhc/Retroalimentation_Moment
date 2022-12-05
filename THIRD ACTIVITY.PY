import requests
import json

from nltk.translate.bleu_score import sentence_bleu

#Document in spanish
es = open("europarl-v7.es-en.es","r",encoding="utf-8") #lecture data of spanish text
doc_es = es.readlines()  # read lines of spanish document
prueb_es = doc_es[0:100] #select 100 cases
print("Document length in Spanish: ",len(prueb_es)) # Check the lenfth of spanish document

#Document in English
en = open("europarl-v7.es-en.en","r",encoding="utf-8") #lecture data of english text
doc_en = en.readlines() # read lines of english document
prueb_en = doc_en[0:100] #select 100 cases
print("Document length in English: ", len(prueb_es)) # Check the length of spanish document


#Deep Translate
def traductor_1(): 
  traducciones_1 =[] #list for save the translations
  url = "https://deep-translate1.p.rapidapi.com/language/translate/v2"
  for line in doc_en:
    payload = {
      "q": line,
      "source": "en",
      "target": "es"
    }
    headers = {
      "content-type": "application/json",
      "X-RapidAPI-Key": "cef5dbb7e7msh89907f251126654p16d8ecjsn6090cb4af60c", #change the key
      "X-RapidAPI-Host": "deep-translate1.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    dictionary_2 = json.loads(response.text)
    trad_1 = (dictionary_2["data"]["translations"]["translatedText"])

    traducciones_1.append(trad_1)#add translations to the list
    
traductor_1()


#translo
def traductor_2():
  traducciones_2 = []#list for save the translations
  url = "https://translo.p.rapidapi.com/api/v3/batch_translate"
  for line in doc_en:
    payload = [
      {
        "from": "en",
        "to": "es",
        "text": line
      }
    ]
    headers = {
      "content-type": "application/json",
      "X-RapidAPI-Key": "cef5dbb7e7msh89907f251126654p16d8ecjsn6090cb4af60c",#change the key
      "X-RapidAPI-Host": "translo.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    dictionary = json.loads(response.text)
    trad_2 = (dictionary["batch_translations"][0]["text"])
    traducciones_2.append(trad_2)#add translations to the list

traductor_2()

#ai translate
def traductor_3():
  traducciones_3 = []#list for save the translations
  url = "https://ai-translate.p.rapidapi.com/translates"
  for line in doc_en:
    payload = {
      "texts": [line] ,
      "tls": ["es"],
      "sl": "en"
    }
    headers = {
      "content-type": "application/json",
      "X-RapidAPI-Key": "cef5dbb7e7msh89907f251126654p16d8ecjsn6090cb4af60c",#change the key
      "X-RapidAPI-Host": "ai-translate.p.rapidapi.com"
    }

    response = requests.request("POST", url, json=payload, headers=headers)
    dictionary_3 = json.loads(response.text)
    trad_3 = (dictionary_3[0]["texts"])

    traducciones_3.append(trad_3)#add translations to the list

traductor_3()

def Scores_translate():# this function claculate the scores for 3 previews translates functions
  for line in doc_es:
    scores_1 =[]
    scores_2 =[]
    scores_3 =[]
    scores_3.append(sentence_bleu(traducciones_3,line))
    scores_2.append(sentence_bleu(traducciones_2, line))
    scores_1.append(sentence_bleu(traducciones_1,line))
  print("Score Third Translate : \n ", scores_3.average())
  print("Score Second Translate : ", scores_2.average())
  print("Score First Translate : ", scores_1.average())

Scores_translate()

#close the documents
doc_en.close()
doc_es.close()
