# NLP-Module-Project

## FIRST TASK:

In this first activity, a model from hugginface was requested to be taken, which had the ability to perform a sentiment analysis so that later data from a txt file was given so that the model could determine if the sentiment that existed in the sentences was positive or negative, for this part we had the dataset which is found in this repository with the name: tiny_movie_reviews_dataset (1).txt

The following shows the results obtained from the tests carried out with the algorithm taken from hugginface
![image](https://user-images.githubusercontent.com/99751023/205555200-d0c28334-9675-46c5-a3a2-86c5636c010a.png)

## SECOND TASK:

In this activity, the objective is based on the retraining of a generic model obtained from the hugginface platform, which allows us to import both datasets and, in this case, models.

As such, a Named entity recognition (NER) model is sought that basically allows us to recognize the keywords of a text, which is represented in the general understanding of the text, this in itself helps us to be able to detect what events They are important within the information.

Likewise, we have also used a hugginface dataset which consists of Wikipedia articles annotated with LOC (location), PER (person), and ORG (organisation) tags in the IOB2 format. This version corresponds to the balanced train, dev, and test splits of Rahimi et al. (2019), which supports 176 of the 282 languages from the original WikiANN corpus.

Below are the metrics obtained from retraining with the new dataset:

![image](https://user-images.githubusercontent.com/99751023/205552913-f1a716ea-11b4-400b-abb6-feeef375d005.png)


## THIRD TASK : 
In this task we seek to use two models already generated in order to compare the translation capabilities.

For this task we have two datasets which have the same content but in different languages (English and Spanish). These documents will help us to determine the quality of the translation made by the models by comparing the result of the translation with the original.

For this reason, we have chosen to take 3 models from the appis page https://rapidapi.com/search/translate, this page allows us to use models for free, although there is a limit derived from the test that is available, although it is more enough for what you want to do. However, it is required to contract the service, although I emphasize that we will use the free mode.

To do this, the following steps must be followed for each of the models since we must place our keys and credentials.

**From the tests that I have done, the key that is generated is the same for the account and for all the services on the page, but I would not rule out that it could change, for that reason I would be careful and generate all 3.**

enter the pages:

https://rapidapi.com/gatzuma/api/deep-translate1/
https://rapidapi.com/haizibinbin-owyntKc0a48/api/ai-translate/pricing
https://rapidapi.com/armangokka/api/translo/

When entering each of the links we can see a sale, where we will have to select pricing
![image](https://user-images.githubusercontent.com/99751023/205537582-60b7b3d3-ee80-40df-b008-e43867a066f2.png)

Next, a screen will be observed where we have to subscribe for free to the most basic plan of each of the apps.
![image](https://user-images.githubusercontent.com/99751023/205538288-f8f63b0f-3bd6-4b7b-bb5b-e7e6da08ba27.png)

Subsequently we will have to return to the first page and select the option python -> requests here we can find our keys

![image](https://user-images.githubusercontent.com/99751023/205538595-32b6c2ff-acd0-4ad7-a2ae-bdde246b830d.png)

![image](https://user-images.githubusercontent.com/99751023/205539232-3265e8a0-c1ee-415e-92f4-18ce521e9bcf.png)

These keys must be replaced as follows:

Deep Learning - translator_1
Translo - translator_2
ai-translate - translator_3

Note to my teacher: I was right, if it's better to use github, I just needed to use it more often, thanks.

### Executing program
- First Activity.py
- Second Activity.py
- Third Activity.py
## Author
Joshua Daniel Hernandez Coronado
