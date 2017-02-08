---
layout: post
title:  "NHL Predictions"
date:   2016-03-01 07:39:18 -0700
categories: [NHL, sport betting, data science]
---
#Features
1. Strength of roster:
- Plus/minus (correlates with goals)
- Power play opportunities

2. Home or on the road

1. History of games against the opponent. More specifically - the frequency of wins/losses. To reflect the importance of the latest games I'll use the weight factor 1.0 for the current season and will reduce weight by 20% for each consecutive season going back, so we'll account for a total of 5 seasons.

2. Parameters reflecting the current season team statistics. About 15 different parameters. Some may be correlated and I will remove them later during the further feature selection.

3. Current win/loss streak. 0.5 for each consecutive win or loss. So L3 is going to be -1.5 and W2 is 1.0.

4. Parameters reflecting the roster strength. For each of non-injured players I'll do a squared sum of each of the available parameters. By doing a squared sum the teams with even rosters will score lower than those having few key scorers, which will provide some variation. Once I have done it for all teams I'll normalize these features by max values in each of the categories.



#Outcome
Single class outcome. We'll assign -1 to a loss, 1 to a win. For OT results we'll use smaller values, i.e. -0.75 for OT loss, 0.75 for OT win. The values are arbitrary and I can later test the sensitivity of the model to these values.




[story-art]: 
