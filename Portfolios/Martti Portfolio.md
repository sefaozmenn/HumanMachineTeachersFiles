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