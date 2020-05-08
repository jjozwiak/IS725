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


nlp = spacy.load('Data/trained_model_v4_5_iter')  # load existing spacy model
print("Loaded custom model")

count = 0

test_data = [
    'The Toronto Blue Jays have agreed to terms with RHP Shun Yamaguchi on a two-year contract US $6,350,000. Yamaguchi had been posted by the Yomiuri Giants of Nippon Professional Baseball. Yamaguchi, 32, pitched 170.0 innings with a 2.91 ERA, 60 walks, and 188 strikeouts over 26 starts for the Yomiuri Giants last season.',
    'The Toronto Blue Jays have agreed to terms on 2020 contracts with INF Brandon Drury US $2,050,000, RHP Ken Giles US $9,600,000, and RHP Matt Shoemaker US $4,200,000.',
    'The Toronto Blue Jays have agreed to terms with free agent INF Travis Shaw on a one-year contract US $4,000,000. To make room on the 40-man roster, INF Richard Ureña has been designated for assignment.',
    'The Toronto Blue Jays have agreed to terms with free agent LHP Hyun-Jin Ryu on a four-year contract US $80,000,000. Ryu, 32, led the Majors with a 2.32 ERA last season and finished second in National League Cy Young voting, going 14-5 while pitching 182.2 innings for the Los Angeles Dodgers.',
    'The Toronto Blue Jays have agreed to terms with free agent RHP Tanner Roark on a two-year contract US $24,000,000. Roark, 33, pitched for the Cincinnati Reds and Oakland Athletics last season, combining for a 10-10 record and a 4.35 ERA across 165.1 innings of work.',
    'The Toronto Blue Jays have agreed to terms with RHP Anthony Bass on a one-year contract US $1,500,000. BASS, 32, went 2-4 with a 3.56 ERA and 43 strikeouts across 48.0 innings pitched for Seattle last season.',
    'The Minnesota Twins announced today that they have signed designated hitter Nelson Cruz to a one-year contract with a club option for the 2020 season. Cruz, 38, played 144 games for the Seattle Mariners last season, hitting .256 (133-for-519) with 18 doubles, 37 home runs, 97 RBI, 70 runs scored, 55 walks and a .342 on-base percentage',
    'The Minnesota Twins announced today that they have signed second baseman Jonathan Schoop to a one-year contract. Schoop, 27, has played six seasons in the major leagues for the Baltimore Orioles (2013-18) and the Milwaukee Brewers (2018).',
    'The Minnesota Twins announced today that they have signed infielder Ronald Torreyes to a one-year contact. Torreyes, 26, played 41 games with the New York Yankees last season, hitting .280 (28-for-100) with seven doubles, one triple, seven RBIs, nine runs scored and a .294 on-base percentage.',
    'The Chicago Bulls have signed guard Antonio Blakeney to an NBA contract. In accordance with team policy, terms of the contract were not disclosed.',
    'The Chicago Bulls announced today the team has claimed guard Antonius Cleveland off waivers. In accordance with team policy, terms of the contract were not announced.',
    'The Chicago Bulls announced today that the team has signed forward Jabari Parker. In accordance with team policy, terms of the contract were not disclosed.',
    'The Chicago Bulls announced today that the team has signed guard Sean Kilpatrick. In a related move, prior to signing Kilpatrick, Chicago waived forward CJ Fair.'
]

for i in range(len(test_data)):
    doc = nlp(test_data[i])
    print("Entities")
    for ent in doc.ents:
        count = count + 1
        print(ent.label_, ent.text)

print(str(count) + ' entities tagged')