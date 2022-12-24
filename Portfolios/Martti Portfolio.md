# Portfolio Minor Applied Data Science<!-- omit in toc -->
### Student name: Martti Groenen<!-- omit in toc -->
### Student number: 19174837<!-- omit in toc -->

This is Martti Groenens portfolio for Applied Data Science. This is where I document my skills and achievements that I acquired during this minor.

# <a id="table-of-contents"></a>Table of Contents <!-- omit in toc -->
- [Obligatory Criteria](#obligatory-criteria)
  - [Datacamp assignments](#datacamp-assignments)
  - [Personal Reflection](#personal-reflection)
  - [Personal Learning Objectives](#personal-learning-objectives)
  - [Evaluation of the group project](#evaluation-of-the-group-project)
- [1. The Project](#1-the-project)
  - [Future work](#future-work)
  - [Conculsions](#conculsions)
  - [Planning](#planning)
- [2. Predictive Analysis](#2-predictive-analysis)
  
- [3. Domain Knowledge](#3-domain-knowledge)
  - [Literature](#literature)
  - [Terminology](#terminology)
- [4. Data Preprocessing](#4-data-preprocessing)
  - [Data preparation](#exploring-and-explanation-of-the-dataset)
  - [Data Visualization](#exploring-and-explanation-of-existing-code)
  - [Data collection](#visualization-and-further-exploration-of-the-data)
  - [Evaluation](#evaluation)
  - [Diagnostics](#diagnostics)
- [5. Communication](#5-communication)
  - [Presentations](#presentations)
  - [Paper](#paper)
  - [Feedback](#feedback)

---

# Obligatory Criteria
## Datacamp assignments

- 5 / 12 / 2022 Datacamp progress

  ![Statement of accomplishment](Datacamp/Martti%20Datacamp.png)


[Back to Table of Contents](#table-of-contents)
## Personal Reflection
I started this minor with having a fair bit of catching up to do. This was because I actually switched minors twice. My group ended up helping me a lot and they showed me what material I missed. I also studied the slides that I had missed the week prior. 

In the beginning I started off as the "coder" of the group. This meant that I did get off track at times, meaning that I chose style over substance. I noticed that style over substance habit of mine was not efficient at all. After a wake-up call from our teachers, I changed my approach. I made sure that I started off my programs small, and gradually and carefully added functionalities ontop of these programs. Using this approach meant that I could methodically add functionalities without breaking a program.

Here are some tasks I worked on per project:

**FoodBoost**
- [Filtered the dataset to drop recipes with nuts]()
- [Applied a PCA model on our case]()
- [Thought of a way to structure data for the final model]()
- [Created a profile generator based on labels]()
- [Created the first Decision Tree Classifier we trained]()

**Containers**
- [Created a GUI for the project using PyGame]()
- [Created an underlying system for moving containers and checking whether the move is legal]()
- [Created a function that converts list of containers to heightmap]()
- [Created multiple environments]()
- Created multiple agents [DQN](), [CNN (with Jesse)]()
- [Changed our observation/action space]()
- []()

- [REMOVE THIS](notebooks/Reconstructie%20Paper%20Model.ipynb)
- [REMOVE THIS](#visualization-and-further-exploration-of-the-data)


Here I will reflect on a situation using STARR:
### **Changing our observation/action space in the container project** <!-- omit in toc -->
#### **Situation** <!-- omit in toc -->
We noticed it took too long for our Deep Q Network to converge on a stable outcome. Sometimes the agent would not even find a stable outcome at all. 

#### **Task** <!-- omit in toc -->
Find a new way for a Deep Q Network to more easily establish associations by changing our observation and action space.

#### **Action** <!-- omit in toc -->
I changed the observation space of the container lot to show a heightmap of the containers. Furthermore I decided to make the actions a 2D matrix, instead of a 3D matrix. This was done since our agent could only put containers ontop of other containers. So there was no real use in having so many unusable actions our agent could take.

#### **Results** <!-- omit in toc -->
This change ended up causing our model to train way faster than it did before.

#### **Reflection** <!-- omit in toc -->
I think this change was a beneficial one. Since making this change showed me that taking a radically different perspective at a problem you're facing can be really advantageous.

[Back to Table of Contents](#table-of-contents)
# 1. The Project

[Back to Table of Contents](#table-of-contents)
# 2. Predictive Models

I have made the following predictive models. I will seperate them per project.
## Project Foodboost
Project foodboost was mainly based around basic machine learning models. During this project I also experimented with Principle Component Analysis (PCA). These predictive models were eventually supposed to recommend recipes to users. 

### PCA
The workflow of my PCA was simple. 
- Firstly, I created a "review" dataset, which included a lot of users. This dataset was generated through matrix multiplications. Due to these matrix multiplications, this dataset ended up having a specific structure. 
- Afterwards, I randomly replaced 60% of the "review" dataset with ``NaN``. 
- Then I replaced the ```NaN```'s with zero's and applied Singular Value Decomposition. Effectively, what this does is it splits the matrix into two seperate matrices.
- Subsequently, I reconstructed the predicted "review" dataset out of the two matrices I created in the prior step. Do note, however, that these values are not close to the ground truth.
- Then I overlayed the known review values (the other 40% of the dataset that was not dropped) over the predicted dataset.
- Upon this mixed dataset I then applied Singular Value Decomposition again. This time, after reconstructing the newly predicted dataset using the two generated matrices, the prediction got a little bit closer to the ground truth.

By iteratively applying those steps (``SVD``, ``Matrix Multiplication``, and ``Overlay original numbers``) we're able to approach the ground truth. Hereby creating an accurate recommendation for a singular user using a huge dataset of reviews from other users. This approach deemed too intricate  for first project. Thus we eventually used a different approach.

### Decision Tree Classifier.

In the new version of this case, we used the Albert Heijn dataset to extract recipe information. This recipe information included ingredients and tags. Using this data, I created a predictive model that was able to train for recommendations for a singular user. To generate the data, I first had to define what tags the user liked, and what tags the user disliked. Using these tags, I was able to randomly pick recipes from those tags. I would extract the ingredients from these recipes, this would be the ``X`` dataset that our predictive model would train on. As for our ``y`` dataset, I set this to 0 for "disliking" a recipe and 1 for "liking" a recipe respectively. Therefore, I generated a dataset with a specific structure in it for our predictive model to train on. My project group decided on having everyone try a different classifier model on the generated dataset. I ended up using ``Decision Tree Classifier`` (as recommended by my group). Using a confusion matrix  to compare this with other models, we were able to deduct that a ``Decision Tree Classifier`` yielded the best results.

## Project Containers

Project containers used a completely different approach to a predictive model. Since this project heavily relied on reinforcement learning, I created my own [``Agent``](), [``Environment``](), and [``Neural Network``]() in pytorch. This environment ended up going through several different iterations. 

### First iteration
The first iteration was heavily based on having a good looking interface. This however, ended up being the thing that slowed the environment down way too much. Besides that, the environment was not created with a 3D matrix in mind, but rather a ``pandas`` dataframe with every entry being a container. This ended up taking up way too much time, and we ended up scrapping this too.

### Second iteration

The second iteration seemed to yield better results. I created a function that would convert a 3D matrix into a 2D heightmap. This heightmap would be interpreted by a basic ``Deep Q Network`` (DQN). I was able to train this model in such a way that it wouldn't attempt to put containers in illegal positions.

### Third iteration

Together with Jesse we created a third iteration. This iteration used roughly the same environment as the second iteration, however, we used a ``Convolutional Neural Network`` (CNN) to interpret the 2D heightmap. Using this approach ended up not working too well. We ended up scrapping this, due to a suggestion I will touch on in the next paragraph.

### Fourth iteration

The Fourth iteration seemed to perform way better than all previous iterations. We got a suggestion from ``Jeroen Vuurens`` that, instead of using a 2D heightmap, or a 3D matrix. We should use something that seperates the rows (X-axis) from the columns (Y-axis) and stacks (Z-axis). Using that clue, I created an environment that would allow the agent to choose between rows. Instead of choosing exactly where to put the container, I only let our agent choose which row to put it in, and the rest was handled by my environment. This approach ended up working really well.  

[Back to Table of Contents](#table-of-contents)
# 3. Domain Knowledge

[Back to Table of Contents](#table-of-contents)
# 4. Data Preprocessing

[Back to Table of Contents](#table-of-contents)
# 5. Communication

## Presentations

During this minor I have presented about 3 to 4 presentations. 

## Paper

## Feedback

- [Obligatory Criteria](#obligatory-criteria)
  - [Datacamp assignments](#datacamp-assignments)
  - [Personal Reflection](#personal-reflection)
  - [Personal Learning Objectives](#personal-learning-objectives)
  - [Evaluation of the group project](#evaluation-of-the-group-project)
- [1. The Project](#1-the-project)
  - [Future work](#future-work)
  - [Conculsions](#conculsions)
  - [Planning](#planning)
- [2. Predictive Analysis](#2-predictive-analysis)
  - [Dichotomous Tree with the OR-Ensemble](#dichotomous-tree-with-the-or-ensemble)
  - [Multinomial Logistic Regression](#multinomial-logistic-regression)
  - [Final OR-ensemble experiments](#final-or-ensemble-experiments)
- [3. Domain Knowledge](#3-domain-knowledge)
  - [Literature](#literature)
  - [Terminology](#terminology)
- [4. Data Preprocessing](#4-data-preprocessing)
  - [Exploring and explanation of the dataset](#exploring-and-explanation-of-the-dataset)
  - [Exploring and explanation of existing code](#exploring-and-explanation-of-existing-code)
  - [Visualization (and further exploration of the data)](#visualization-and-further-exploration-of-the-data)
- [5. Communication](#5-communication)
  - [Presentations](#presentations)
  - [Paper](#paper)
  - [Feedback](#feedback)