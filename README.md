# DISRUPTIVE BEHAVIOR AS A FUNCTION OF SCREEN & OUTDOOR HABITS

## Overview
This project explores the data collected by Deakin University in Melbourne. The study found relationships between screen time, outdoor time, and certain behavioral outcomes.

Read on to see how I used supervised learning to see if this data can be taken a step further to actually predict disruptive behavior in children.

The study can be read in PLOS [here]([link-to-PLOS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0193700)), and the original data is available from Deakin University [here]([link-to-Deakin-University](https://dro.deakin.edu.au/view/DU:30106962)).

## Study Methods
575 Mothers from diverse neighborhoods in Melbourne, Australia were surveyed about their child between 2-5. They were asked to report:

- Child's age, gender, and Mother's education level (<10 years, 11-13 years, or 14+ years of school)
- Child's hours per day watching TV and using the computer
- Child's hours per day outside
- Whether they consider the child disabled

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/30a0c4b4-0bbe-4850-8c89-b80741888269)

Additionally, they reported their child's social skills using the Adaptive Social Behaviour Inventory. This involves answering 30 questions on 3 point Likert scales (Always/Sometimes/Never) related to their child's behavior. Of those questions:

- 13 questions targeted the child's "expressiveness" (e.g. joins play, is open and direct, etc)
- 10 questions targeted the child's "compliance" (e.g. cooperates, is calm and easy going, etc)
- 7 questions targeted the child's "disruptive" behavior (e.g. teases, bullies, etc)

Each of these 3 sub-scales was summed to create 3 scores describing the behavior of that child.

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/4706bb29-ba8a-4416-a51f-577bf6631131)

## Limitations
Two limitations of this study should be considered:

- Self Reporting: This study relies on mothers' ability to make objective assessments about their child's behavior
- Selection: The study authors made reasonable efforts to sample the population of mothers fairly, but it's still likely a bias sample; good parents may be more likely to agree to participate in a parenting study than indifferent parents, or parents with a difficult socioeconomic status. My suspicion is further heightened by the Mother's education results. In most developed countries, the percentage of adults with tertiary education is 20-50%, not ~75% as represented in this study.

## Full Feature Set
The study authors also used the Australian Government's recommendations for children's screen and outdoor time. Here is the final feature set available, including our target, "Disrupt".

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/e2d0f1c6-4dae-4fa2-b367-e023ff3aac6b)

## Scenario
Let's bring this to life with a scenario: suppose a school administrator is trying to prevent classrooms from having too many disruptive students, so they are trying to identify which incoming students are likely to be disruptive and be ahead of the problem.

Suppose we decide that 13 is the cutoff, so the goal is to predict which students will have a disrupt score of 13 or higher:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/ffb0a636-4331-4a46-a982-6a04255567b6)
- 33/575 students qualify as "disruptive"

## Baseline Model
We'll use a Logistic Regression to classify the students as either "Behaved" or "Disruptive". Without any further work, let's examine the output of a cross-validated model and see how it does:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/cf6a0ed3-c91c-44a2-9cce-4dce30740895)

The Confusion Matrix (the left graph) tells us that while all 542 Behaved students were classified correctly, only 1 of the 33 Disruptive students were identified.

The Distributions (the center graph) show us why. scikit-learn's Logistic Regressor defaults to a .5 decision threshold, i.e. it will only predict "Disrupt" for a student if it calculates .5 or higher for that student. In this case, that's just one little red bar. For a better understanding of how to interpret these distributions in the context of the ROC curve, check out this visualization.

Unlike the other two graphs, the ROC curve (the right graph) is not a description of one model, but of an entire threshold of model decision points. Right now, the model is on the far bottom left: Our False Positive Rate is zero (0/542) and our True Positive Rate is .03 (1/33). This ROC curve tells us the tradeoff we could make: by "moving up the curve", we could identify more disruptive students (i.e. increase True Positives) but we would also falsely accuse behaved students more often (i.e. increase False Positives).

This is a bad model. It's F1 Score, a score which factors both precision and recall, is only .06.

This is a classic Class Imbalance problem! our Logistic regressor converges on a solution that favors Behaved students, just because there are so many more of them. We need the model to view the two groups more equally.

## Fixing Class Imbalance with weighting
One option is to weight the data points in proportion to their representation in the dataset. In this case, the error on a Disruptive student counts 16x more (542/33) than an error on a Behaved student. Let's see how that does:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/68523890-8082-4729-ad88-c698cdb5fabc)

We've still only identified 5/33 Disruptive students, and what's worse, we've now falsely accused 42 Behaved students of being disruptive. The Distribution of predictions are all very close to .5, which indicates that this model is generally unsure of everything. Balanced class weighting may have been an overreaction.

Despite poor results, the F1 Score increased to 0.12 in this model.

Rather than guessing, let's search for the right class weight to use by plotting F1 Scores for all class weight options:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/d5e3502c-e900-49e4-8e08-862952e47f4f)

Although F1-score peaks at 7, that appears to be a volatile region. 19 is a much more stable peak, and is also closer to the true class ratio in our data, which is 16:1. Let's try class weights of 19:1:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/83534dac-ed25-4338-8ca0-67ed4e9d2dcf)

Our F1 Score has increased to .20. For the first time, the model is erring on the side of predicting "Disrupt" rather than not. This is how the distributions should look: some samples near 0 (definitely Behaved), some near 1 (Definitely Disruptive), and some near the middle. The goal of a good model is to reduce the overlap between red and green as much as possible.

At this point, we could tell the school something like this, based on the ROC curve above and choosing the decision threshold circled in red:

"Our current model could correctly identify ~85% of disruptive students; however, it would also result in almost 40% of well-behaved students being mistakenly labelled as disruptive".
Obviously, this is not a usable solution - there are way too many false positives.

## Removing Collinearity
Continuous variables, like TV time or Outdoor Time, are correlated with each other. Having correlated features violates an assumption of many generalized linear models, including Logistic Regression. In this case Principal Components Analysis can be used to remove multicollinearity without losing any information, but at the expense of interpretability.

Principal Components Analysis deserves its own post (and will get one, coming soon!), but the short version is that PCA redefines the basis system we’re using to measure our data. The new basis, or "coordinates", for each point will still describe all the variability in the original dataset, but the dimensions will be orthogonal to each other so there's no multicollinearity.

PCA works best for moderately correlated features. In this case, the following groups were respectively transformed to components:

- Age, Comply, Express
- tvTime, cpuTime, outdoorTime

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/42c4ec17-16c4-4d19-9872-352bf895e4f0)

With these new features, we were able to identify one additional Disruptive student successfully. This may not seem like much, but given that we did that with no new information, it's pretty neat!

## More Boolean Features and PCA Components
The last feature engineering step that improved outcomes was to create some additional binary variables.

- Whether or not the child's screen time measurements were BOTH (TV and Computer) over the average
- Whether their expressiveness score was above average
- Whether their compliance score was above average
- Mother's education (either 14+ years or not)

These were combined with the other binary features in the dataset:

- Disability
- Screen Time Requirement Met
- Physical Requirement Met
- Gender

All 8 of these were then transformed into PCA components.

PCA on binary variables is a different beast; I'll have an article about this process up soon. One cool difference is that the binary nature of the original features enables easy plotting of the first 2 PCA components, and interpretation of what compose them. Below is the same plot 8 times: the first 2 PCA components. Each plot has a different color scheme highlighting one of the original 8 features listed above. You can actually see which features compose the first two components!

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/e5641ce2-f1a2-4815-a8ba-784fdd77b46d)

While PCA components are normally not very interpretable, in this case, we can actually see that disability explains a lot of the variance in this data, while Mother's Education does not (at least not in the first 2 components).

Anyway - after adding the Binary features, I plotted F1 scores again, and this time the best score used weights of 7:1, and also appeared pretty stable:

![image](https://github.com/Monish24/distruption_prediction/assets/54630644/d169df96-8887-4a66-8394-5db8109b9ab5)

## Final Model
![image](https://github.com/Monish24/distruption_prediction/assets/54630644/8e3be612-9bd2-47be-82b4-d3fcf616685d)

This model has an F1 score of .24, which is the best I could do with this data. Here's a summary of the models tried, including a couple I didn't highlight here:
![image](https://github.com/Monish24/distruption_prediction/assets/54630644/300dddbf-d45b-46d6-a93d-4e7abc0e19a7)

## Summary
Screen time and outdoor time clearly have correlations with behavior, as was identified in the original study. Additionally, predictive methods could be used to identify extreme outliers. In general, however, more data would need to be collected to make confident predictions about a disruptive behavior.


