import pandas as pd
# random lines some good others bad
dataset = pd.read_csv('Random-texts.csv') 
dataset.head(10)

import spacy
# nlp lib to handle . I'm folowing bag of words approach
nlp = spacy.blank("en")  #model is called nlp

# text categroizer wit std settings
textcat = nlp.create_pipe("textcat", config={
                "exclusive_classes": True,
                "architecture": "bow"}) #bow = bag of words

nlp.add_pipe(textcat) #add textcat to nlp
textcat.add_label("Good")#add labels
textcat.add_label("Bad")
1
train_texts = dataset['Line'].values   #training a Text Categorizer Model
train_labels = [{'keys': {'Bad': x == 'Bad', 'Good': x == 'Good'}} 
                for x in dataset['Good/Bad']]
from spacy.util import minibatch
optimizer = nlp.begin_training() #create optmizer to be used by spacy to update the model

batches = minibatch(train_data, size=8)   #spacy provides minibatch fn

for batch in batches:
    texts, labels = zip(*batch)
    nlp.update(texts, labels, sgd=optimizer)
# i mentioned all lines to be predicted in a 'texts' array
Lines = ["I look awesome"]
docs = [nlp.tokenizer(text) for text in Lines]
    
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)  #prob score for both classes (Good/bad)
print(scores)

predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])