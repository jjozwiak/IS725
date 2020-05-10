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


nlp = spacy.load('Data/trained_model_v2_200_iter')  # load existing spacy model
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
    'The Chicago Bulls announced today that the team has signed guard Sean Kilpatrick. In a related move, prior to signing Kilpatrick, Chicago waived forward CJ Fair.',
    'Cincinnati Reds President of Baseball Operations Dick Williams today announced the acquisition of RHP Trevor Bauer from the Cleveland Indians as part of a 3-team trade in which the Indians received from the Reds OF Yasiel Puig and LHP Scott Moss and from the San Diego Padres OF Franmil Reyes, LHP Logan Allen and IF Victor Nova, while the Padres received from the Reds OF Taylor Trammell (truh-MELL). Bauer, 28, this season leads the Major Leagues with 156.2 innings pitched while ranking fifth in the majors and fourth in the American League with 185 strikeouts. In his 24 starts for the Indians, he went 9-8 with a 3.79 ERA.',
    'Cincinnati Reds President of Baseball Operations Dick Williams today announced the acquisition of RHP Tanner Roark (RO-arc) from the Washington Nationals in exchange for RHP Tanner Rainey. Roark, 32, has spent his entire Major League career with the Nationals, compiling a 64-54 record and 3.59 ERA over 6 seasons, 141 starts and 41 relief appearances. In 2018, he went 9-15 with a 4.34 ERA in 30 starts and 1 relief appearance and ranked 12th in the National League with 17 quality starts.',
    'The Cincinnati Reds and RHP Raisel Iglesias have agreed to terms on a 3-year contract through the 2021 season.',
    'Cincinnati Reds President of Baseball Operations Dick Williams today announced the acquisitions of RHP Lucas Sims, RHP Matt Wisler and OF Preston Tucker from the Atlanta Braves in exchange for LF Adam Duvall. Sims, 24, has been one of Atlanta\'s top prospects since he was selected in the first round (21st overall) of the 2012 first-year player draft.',
    'The Texas Rangers today announced that the club has agreed to terms with right-handed pitcher Taylor Jungmann (YOUNG-men ) on a minor league contract. Jungmann, 30, has spent the past two seasons with the Yomiuri Giants in Japan, pitching in both Nippon Professional Baseball (NPB) as well as appearing for the club\'s Japan Eastern League affiliate.',
    'The Texas Rangers today announced the signing of right-handed pitcher Cody Allen to a 2020 minor league contract with an invitation to Major League spring training camp. Allen, 31, went 0-2 with 4 saves and a 6.26 ERA over 25 relief appearances with the Los Angeles Angels.',
    'The Texas Rangers today announced that the club has signed first baseman Greg Bird to a minor league contract with an invitation to Major League spring training camp. Bird, 27, batted .171 (6-35) with one home run and one RBI in 10 games for the Yankees.',
    'Texas Rangers today announced that the club has signed third baseman Matt Duffy and right-handed pitcher Derek Law to minor league contracts with an invitations to Major League spring training camp. The 29-year-old Duffy has spent the past four seasons with the Tampa Bay Rays.',
    'The Texas Rangers announced this afternoon that the club has acquired first baseman/outfielder Sam Travis from the Boston Red Sox in exchange for left-handed pitcher Jeffrey Springs. The 26-year-old Travis was designated for assignment by the Red Sox on January 2 and assigned outright to Triple-A Pawtucket.',
    'The Texas Rangers today announced that the club has signed free agent catcher Robinson Chirinos and third baseman Todd Frazier to one-year contracts covering the 2020 season with club options for the 2021 season.',
    'The Texas Rangers today announced that the club has acquired outfielder Adolís García (pronounced ah-DOH-lees) from the St. Louis Cardinals in exchange for cash considerations. To make room for García on the Major League roster, right-handed pitcher Jimmy Herget has been designated for assignment.',
    'The Texas Rangers tonight announced that the club has acquired outfielder Steele Walker from the Chicago White Sox in exchange for outfielder Nomar Mazara. Walker, 23, completed his first full professional season in 2019, combining for a .284 average with 10 homers and 62 RBI.',
    'The Texas Rangers today announced that the club has signed free agent left-handed pitcher Joely Rodríguez to a two-year contract covering the 2020 and 2021 seasons, along with a club option for the 2022 season. Financial terms were not disclosed. Rodriguez, 28, has spent the last two seasons playing in Japan for the Chunichi Dragons of Nippon Professional Baseball, going 3-7 with a 1.85 ERA over 90 relief appearances.',
    'The Texas Rangers today announced that the club has acquired right-handed pitcher Corey Kluber, along with cash considerations, in exchange for outfielder Delino DeShields and right-handed pitcher Emmanuel Clase. The 33-year-old Kluber was selected as the American League Cy Young Award winner in both 2014 and 2017, and he also earned A.L. All-Star honors in three consecutive seasons from 2016-18.',
    'The Texas Rangers tonight announced that the club has signed free agent right-handed pitcher Jordan Lyles to a two-year contract covering the 2020 and 2021 seasons. The team has also signed free agent left-handed pitcher Jeffrey Springs to a Major League contract. Lyles, 29, combined to post a 12-8 record with a 4.15 ERA over 28 games/starts last season for Pittsburgh and Milwaukee, recording career bests in wins, starts, strikeouts (146), and strikeouts per 9 innings (9.3).',
    'The Texas Rangers today formally announced the signing of free agent right-handed pitcher Kyle Gibson to a three-year contract covering the 2020, 2021, and 2022 seasons. Financial terms were not disclosed. The 32-year-old veteran compiled a 13-7 record with a 4.84 ERA and 1.444 WHIP figure over 34 games/29 starts for the American League Central Division champion Minnesota Twins in 2019.',
    'The Texas Rangers today announced that the club has acquired right-handed pitcher Nick Goody on a release waivers claim from the Cleveland Indians. Goody, 28, was designated for assignment by the Indians on November 20 and placed on unconditional release waivers on November 22.',
    'The San Francisco Giants announced today that they have signed infielder Wilmer Flores to a two-year contract worth $6.25 million. Flores, 28, spent the 2019 season with the division-rival Arizona Diamondbacks and batted .317 with 18 doubles, nine home runs and 37 RBI in 285 plate appearances with Arizona and had a batting line of .337/.367/.615 over 109 plate appearances against left-handed pitching last year.',
    'The San Francisco Giants announced today that they have signed outfielder Hunter Pence to a one-year $3,000,000 contract for the 2020 season. The 37-year-old was the 2019 A.L. Sporting News Comeback Player of the Year after he posted a .297/.358/.552 batting line with 18 home runs, 17 doubles, a triple, 59 RBI and six stolen bases in 316 plate appearances with his hometown Texas Rangers.',
    'The San Francisco Giants announced today that they have signed right-handed pitcher Kevin Gausman to a one-year contract for 2020 worth $9 million. Gausman can make up to an additional $1 million in performance bonuses: $250,000 each for 18, 22, 26 and 30 games started as a pitcher. Gausman, 29, split last season with the Braves and Reds combining for a 3-9 record and a 5.72 ERA in 31 games (17 starts). He began the season with Atlanta, going 3-7 with a 6.19 ERA in 16 starts before being claimed by the Reds in August and going 0-2 with a 4.03 ERA in 15 games (one start).',
]

for i in range(len(test_data)):
    doc = nlp(test_data[i])
    print("Entities")
    for ent in doc.ents:
        count = count + 1
        print(ent.label_, ent.text)

print(str(count) + ' entities tagged')