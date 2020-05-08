#!/usr/bin/env python
# coding: utf8

# Training additional entity types using spaCy
from __future__ import unicode_literals, print_function
import pickle
import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# New entity labels
# Specify the new entity labels which you want to add here
LABEL = ['I_new_team_city', 'B_new_team_city', 'I_player_position', 'B_player_position', 'I_player_name', 'B_player_name', 'I_player_age', 'B_player_age', 'I_new_team', 'B_new_team', 'I_contract_amount', 'B_contract_amount', 'O']

"""
new_team_city = new team city
player_position = player position
player_name = player name
player_age = player age
new_team = new team
contract_amount = contract amount

"""
# Loading training data 
with open ('Data/proj_spacy', 'rb') as fp:
    TRAIN_DATA = pickle.load(fp)

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int))

def main(model=None, new_model_name='new_model', output_dir= "Data/models", n_iter=2):
    """Setting up the pipeline and entity recognizer, and training the new entity."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spacy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe('ner')

    for i in LABEL:
        ner.add_label(i)   # Add new entity labels to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.entity.create_optimizer()

    # Get names of other pipes to disable them during training to train only NER
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            batches = minibatch(TRAIN_DATA, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35,
                           losses=losses)
            print('Losses', losses)

    # Test the trained model
    #test_text = 'The Atlanta Braves today signed infielder Jason Jozwiak to a one-year contract worth $1 million. Jozwiak, 35 has been an outstanding player.'
    test_text = 'The Toronto Blue Jays have agreed to terms with RHP Shun Yamaguchi on a two-year contract (US $6,350,000). Yamaguchi had been posted by the Yomiuri Giants of Nippon Professional Baseball. Yamaguchi, 32, pitched 170.0 innings with a 2.91 ERA, 60 walks, and 188 strikeouts over 26 starts for the Yomiuri Giants last season.'
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # Save model 
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # Test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == '__main__':
    plac.call(main)
