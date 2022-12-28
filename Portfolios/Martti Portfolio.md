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

### **STARR:** 
#### **Situation**


#### **Task**


#### **Action**


#### **Results**


#### **Reflection**

## Learning Objectives
For this project I set the following learning objectives when I started:
- Develop a strong foundation in programming and data analysis techniques, including proficiency in Python and experience with statistical and machine learning methods. 
- Be able to identify and solve real-world problems using data-driven approaches, including the ability to formulate research questions, collect and clean data, apply appropriate statistical and machine learning techniques, and communicate findings effectively.
- Gain hands-on experience with a variety of data science tools and technologies, including database management systems, data visualization software, and machine learning libraries.

Through the Applied Data Science minor program, I was able to develop a strong foundation in Python programming and data analysis techniques. I had some experience with Python beforehand, but I was able to deepen my understanding and skills through coursework and hands-on projects. I applied the skills I learned in the lectures in a real-world setting through a project using Albert Heijn recipe data and tags to create a recipe recommendation system for people with specific tastes. I gained hands-on experience with various data science tools and technologies, including Pandas, Matplotlib, and Numpy through Datacamp.

### **STARR:** 
#### **Situation**


#### **Task**


#### **Action**


#### **Results**


#### **Reflection**


## Group Reflection/Evaluation
I really enjoyed working with my project group. This has been one of the better groups I've done projects with. However, this does not mean this entire project went without a hitch. In the sections below, I will be evaluating each project member seperately.

### Jesse
Starting with Jesse. From what I've seen from Jesse, he knows how to dive deep into a subject and get into the nitty gritty. This means really getting to know *how* something works before actually starting to realise the product. This is a different approach to mine, but proved effective in combination with mine. 

### Eric
Whenever I'd code with Eric, he'd have a lot of good and calculated insights into what errors I could be making. Furthermore, he also was the main note taker of our project group, this made accessing feedback we got during presentations or meetings way easier and definitely sped up development.

### Joanne
Joanne mainly kept the group together. Her role in this project was mainly being a chairwoman. She'd put us back on topic if we'd drift off topic in a meeting. Furthermore she'd also create really well layed out agendas. I also noticed she made a lot of progress when it comes to her knowledge in applied data science. Especially when we parallelized our work was when she made a lot of progress.

### Sefa
Since Sefa does the same study I do (HBO-ICT) he did have a little bit more coding experience to other teammates. However, complex concepts like reinforcement learning or PCA were a little bit harder to understand for him. Mainly since this is not part of his study. Besides this, I did notice that he made a lot of progress in the latter part of the project. He ended up creating his own working reinforcement learning model.

### Ayrton
Overall as a person, I'd say that Ayrton is a really quiet person. Because of this, it's hard for me to gauge how much progress he made. In a meeting, whenever I'd ask Ayrton about his thoughts on the matter at hand, he'd usually reply "I'm okay with anything". This was not really too useful for us, since we needed actual opinions on the matter at hand. Eventually, we tasked Ayrton to create a reinforcement learning model in about 3 to 4 weeks. He ended up missing that deadline and getting a second chance. After the second deadline passed, he had a working Q-Learning model, however, it did not seem to be able to explain this model at all. This may be because he felt pressured. So in general, I'm not sure how much progress Ayrton made.

### **STARR: Project group work imbalance** 
#### **Situation**
The level of my coding skills was far above my teammates. This resulted in me making a lot of progress, while my teammates couldn't keep up.

#### **Task**
Adopt a method that enables my teammates to make progress without hindering my own ability to progress and without degrading the final product.

#### **Action**
Jesse brought up a good work method, we ended up having everyone make their own version of the program/model. Effectively having everyone work in parallel.

#### **Results**
This meant that we wouldn't get stopped or held back by other teammates, and thus we could take things at our own pace.

#### **Reflection**
This method proved effective since most teammates now know the inns and outs of their own model. Ontop of that, teammates were able to approach other teammates in case they had any questions.

## Contributions

Here are some tasks I contributed on per project:

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
- [Changed our observation/action space (twice)]()
- []()

- [REMOVE THIS](notebooks/Reconstructie%20Paper%20Model.ipynb)
- [REMOVE THIS](#visualization-and-further-exploration-of-the-data)

Here I will reflect on a situation using STARR:

### **STARR: Changing our observation/action space in the container project** 
#### **Situation** 
I noticed it took too long for our Deep Q Network to converge on a stable outcome. Usually the agent would not even find a stable outcome at all. 

#### **Task** 
Find a new way for a Deep Q Network to more easily establish associations by changing our observation and action space.

#### **Action** 
I changed the observation space of the container lot to show a heightmap of the containers. Furthermore I decided to make the actions a 2D matrix, instead of a 3D matrix. This was done since our agent could only put containers ontop of other containers. So there was no real use in having so many unusable actions our agent could take.

#### **Results** 
This change ended up causing our model to train way faster than it did before.

#### **Reflection** 
I think this change was a beneficial one. Since making this change showed me that taking a radically different perspective at a problem you're facing can be really advantageous.




[Back to Table of Contents](#table-of-contents)
# 1. The Project
\<Insert project text>

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

The Fourth iteration seemed to perform way better than all previous iterations. We got a suggestion from ``Jeroen Vuurens`` that, instead of using a 2D heightmap, or a 3D matrix. We should use something that seperates the rows (X-axis) from the columns (Y-axis) and stacks (Z-axis). Using that clue, I created an environment that would allow the agent to choose between rows. Instead of choosing exactly where to put the container, I only let our agent choose which row to put it in, and the rest was handled by my environment. This approach ended up working really well. The training graph of this final approach is shown in the graph below.

  ![Graph of Fourth Iteration](images/FourthIterGraph.png)

There are two features to this graph that need some explaining. First off the theoretical maximum is based on the absolute maximum score that our environment can put out. Our model will never be able to score beyond that value. Secondly, these 79200 games took around one and a half hours to complete. This may seem long, however, this does not impact the eventual prediction speed.

[Back to Table of Contents](#table-of-contents)
# 3. Domain Knowledge
\<Insert Domain knowledge>

[Back to Table of Contents](#table-of-contents)
# 4. Data Preprocessing
\<Insert Data Preprocessing>
[Back to Table of Contents](#table-of-contents)
# 5. Communication

## Presentations

During this minor I have presented about 3 to 4 presentations. TODO: What were they abt and a link to the presentations 

## Paper
For our paper I wrote ``insert part here``: link to paper

