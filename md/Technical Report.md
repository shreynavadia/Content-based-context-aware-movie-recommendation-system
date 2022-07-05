```python
display(_image(filename="header.PNG"))
```


![png](output_0_0.png)


<h2><font color='darkorange'> Executive Summary </font></h2>

<p style="text-align:justify">The growth of digital movie streaming over the years paved way to the creation of a huge amount of data containing ratings of users to movies, content metadata such as genre and title of movies, context of the user like gender age, and occupations$^1$. Recommender systems are helpful in trying to predict the preference or the possible ratings of users for items that they haven't tried yet. In this technical report the team aims to answer How might we provide effective movie recommendations using several user specific features and movie features?</p>
<p style="text-align:justify">Using the MovieLens 1M dataset that contains movies from 1995 to 2015 and has million of anonymous ratings from six thousand users who rated at least twenty movies we explored the different characteristics of genre, age, occupation, and gender in terms of count and its average ratings. We found interesting insights such as older group tend to rate higher compared to the younger ones, demographics of the Movielens users, and drama as one of the most popular genre in the MovieLens. These content metadata and context of the users are used as features to our recommender system models.</p>
<p style="text-align:justify">The team developed content-based recommender system using genre and title as the features. We preprocessed the text using NLTK and converted it to machine readable representation using TF-idf. The TF-idf features are used to train our model based recommender system. The results that we got are similar to the user preference in terms of genre.</p>
<p style="text-align:justify">The team also developed several Context-Aware recommender system using genre, age group, and occupation. the recommendations made are more influenced by the filters set instead of the viewing or rating history. Using pre-filtering method, the model did poorly for some of the context used due to its effect on initial filtering of data. For both of the post filtering and contextual modeling, the results that we got performed well even on having multiple context at a time.  The team also developed a combination of content based and context aware models by getting its weighted sum in order to get a more specific recommendation based on the content and context of the user.</p>
<p style="text-align:justify">For future studies, we suggested to use a larger dataset, try other models, and perform evaluation metrics to further validate the results.</p>


```python
import re
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from matplotlib import pyplot as plt
from tqdm import tqdm
from IPython.display import HTML
from IPython.display import Image as _image

tqdm.pandas()

import datetime

import pyspark
from pyspark.ml import Pipeline
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.regression import FMRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import (StringIndexer, VectorAssembler, VectorIndexer,
                                OneHotEncoder)

from surprise import Reader, Dataset, KNNWithMeans

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Download nltk requirements
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')


ML_COLORS = ['darkorange']

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<style>
.output_png {
    display: table-cell;
    text-align: center;
    horizontal-align: middle;
    vertical-align: middle;
    margin:auto;
}

tbody, thead {
    margin-left:100px;
}

</style>
<form action="javascript:code_toggle()"><input type="submit"
value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<style>
.output_png {
    display: table-cell;
    text-align: center;
    horizontal-align: middle;
    vertical-align: middle;
    margin:auto;
}

tbody, thead {
    margin-left:100px;
}

</style>
<form action="javascript:code_toggle()"><input type="submit"
value="Click here to toggle on/off the raw code."></form>




```python
# define the context aware recommender system
# code revised from prof leo's code. thank you!

class ContextAwareRS:
    """Base class for context-aware recommender systems"""

    def __init__(self, data, context_vars, rating_scale, sim_options=None):
        """Initialize the context aware recommender system given the
        ratings data and context variables
        """
        # Initialize static variables
        self.data = data
        self.context_vars = context_vars
        self.rating_scale = rating_scale
        if sim_options is None:
            self.sim_options = {'name': 'pearson', 'user_based': False,
                                'min_support': 0, 'shrinkage': 0}
        else:
            self.sim_options = sim_options

        # Initialize recsys model
        self.model = None
        
class ContextPreFiltering(ContextAwareRS):
    """Class for implementing context prefiltering algorithm with a
    neighborhood-based collaborative filtering algorithm
    """

    def __init__(
        self, data, context_vars, rating_scale, sim_options=None, k=None):
        """Initialize the context-prefiltering model"""
        super().__init__(data, context_vars, sim_options)
        if k is None:
            self.k = 5
        else:
            self.k = k

        # Initialize current data - this will be the 'filtered' data
        # based on context
        self.cur_data = None

    def fit(self, context_var_values):
        """Fit a context-aware recsys given context_vars_value"""
        # Get current data based on given context values
        cur_data = self.data.copy()
        for col, val in zip(self.context_vars, context_var_values):
            # If a None value is given, skip that column
            if val is not None:
                cur_data = cur_data.loc[cur_data.loc[:, col] == val]

        # Append to the CPF object for reference later
        self.cur_data = cur_data

        # Perform KNNWithMeans training
        reader = Reader(self.rating_scale)
        dataset = Dataset.load_from_df(self.cur_data.iloc[:, :3], reader)
        knn = KNNWithMeans(k=self.k, sim_options=self.sim_options)
        knn.fit(dataset.build_full_trainset())

        # Save model to object
        self.model = knn

    def show_top_k(self, user_id, top_k=20):
        """Return the context-aware top-k recommendations for user"""
        # Retrieve items not seen by the user
        seen_items = (self.cur_data[self.cur_data.iloc[:, 0] == user_id]
                      .iloc[:, 1].unique())
        #if seen_items == []:
        #    print('no seen movies, so recomms will have same rating')
        #else:
        #    pass
        unseen_items = (
            self.cur_data[~self.cur_data.iloc[:, 1].isin(seen_items)]
            .iloc[:, 1].unique()
        )

        #return seen_items
        
        # Generate predictions
        predictions = [self.model.predict(user_id, item)
                       for item in unseen_items]

        # Sort predictions based on estimated rating
        return [(prediction.iid, prediction.est) 
                for prediction in sorted(predictions, 
                                         key=lambda x: -x.est)][:top_k]

class ContextPostFiltering(ContextAwareRS):
    """Class for implementing context postfiltering algorithm with a
    neighborhood-based collaborative filtering algorithm
    """

    def __init__(
        self, data, context_vars, rating_scale, sim_options=None, k=None):
        """Initialize the context-prefiltering model"""
        super().__init__(data, context_vars, sim_options)
        if k is None:
            self.k = 5
        else:
            self.k = k

        # Initialize filter matrix. This will be used to filter the
        # resulting ratings at prediction time
        self.cur_data = None
        self.filter_matrix = None

    def fit(self, context_var_values, min_rating=3):
        """Fit a context-aware recsys given context_vars_value"""
        # Get current data based on given context values
        cur_data = self.data.copy()
        for col, val in zip(self.context_vars, context_var_values):
            # If a None value is given, skip that column
            if val is not None:
                cur_data = cur_data.loc[cur_data.loc[:, col] == val]
        self.cur_data =cur_data

        # Solve for the filter matrix. Here, we take P(*, i, c) as the
        # ratio between users who rate the movie with
        # stars > min_rating, over the total number of users.
        filter_matrix = cur_data.groupby('movieid').rating.apply(
            lambda x: (x >= min_rating).sum() / len(x)
        )
        self.filter_matrix = filter_matrix

        # Aggregate data into two-dimensional ratings matrix
        aggregate_data = (self.data.groupby(self.data.columns[:2].tolist())
                          .rating.mean().reset_index())
        
        # Perform KNNWithMeans training
        reader = Reader(self.rating_scale)
        dataset = Dataset.load_from_df(aggregate_data, reader)
        knn = KNNWithMeans(k=self.k, sim_options=self.sim_options)
        knn.fit(dataset.build_full_trainset())

        # Save model to object
        self.model = knn

    def show_top_k(self, user_id, top_k=20):
        """Return the context-aware top-k recommendations for user"""
        # Retrieve items not seen by the user
        seen_items = (self.cur_data[self.cur_data.iloc[:, 0] == user_id]
                      .iloc[:, 1].unique())
        if seen_items == []:
            print('no seen movies, so recomms will have same rating')
        else:
            pass
        unseen_items = (
            self.cur_data[~self.cur_data.iloc[:, 1].isin(seen_items)]
            .iloc[:, 1].unique()
        )

        # Generate predictions
        predictions = pd.DataFrame([self.model.predict(user_id, item)
                                    for item in unseen_items])
        predictions = predictions.set_index(predictions.columns[1]).est

        # Perform the post-filtering process
        predictions = ((predictions * self.filter_matrix)
                       .sort_values(ascending=False))

        # Sort predictions based on estimated rating
        return predictions[:top_k]
    
class ContextualModeling(ContextAwareRS):
    """Class for implmenting contextual modeling using factorization
    machines
    """

    def __init__(self, data, context_vars):
        """Initialize the contextual modeling recommender system"""
        super().__init__(data, context_vars, None)

    def fit(self):
        """Fit given the context variable values"""
        # Set the pipeline for the Factorization machines
        pipe = Pipeline(stages=[
            # Create string indices for each string variable
            StringIndexer(
                inputCols=['userid', 'movieid'],
                outputCols=['useridIndex', 'movieidIndex']),

            # One hot encode variables
            OneHotEncoder(
                inputCols=['useridIndex', 'movieidIndex', 'age', 
                           'occupations'],
                outputCols=['userID', 'movieID', 'ageID', 'occID']),

            # Assemble onto one vector
            VectorAssembler(
                inputCols=['userID', 'movieID', 'ageID', 'gender', 'occID'],
                outputCol='features', handleInvalid='skip'),

            # Train a FM model
            FMRegressor(
                featuresCol='features', labelCol='rating', stepSize=0.001)
        ])

        # Create spark data frame then fit the model to it
        sdf = spark.createDataFrame(self.data)
        model = pipe.fit(sdf)

        # Save the model to object
        self.model = model

    def show_top_k(self, user_id, context_var_values, top_k=20):
        """Return the context-aware top-k recommendations for user"""
        # Get current data based on given context values
        cur_data = self.data.copy()
        for col, val in zip(self.context_vars, context_var_values):
            # If a None value is given, skip that column
            if val is not None:
                cur_data = cur_data.loc[cur_data.loc[:, col] == val]

        # Retrieve items not seen by the user
        seen_items = (cur_data[cur_data.iloc[:, 0] == user_id].iloc[:, 1]
                      .unique())
        unseen_items = (cur_data[~cur_data.iloc[:, 1].isin(seen_items)]
                        .iloc[:, 1].unique())

        # Setup test data frame
        test_df = pd.DataFrame({'movieid': unseen_items})
        test_df.loc[:, self.data.columns[0]] = user_id
        test_df.loc[:, self.context_vars[0]] = context_var_values[0]
        test_df.loc[:, self.context_vars[1]] = context_var_values[1]
        test_df.loc[:, self.context_vars[2]] = context_var_values[2]
        test_sdf = spark.createDataFrame(test_df)

        # Get predictions
        predictions = (self.model.transform(test_sdf)
                       .select(['movieid', 'prediction'])
                       .toPandas().sort_values('prediction', ascending=False))

        # Sort predictions based on estimated rating
        return predictions[:top_k]
```

<h2><font color='darkorange'> I. Introduction </font></h2>

<p style="text-align:justify">Movies have now become more accessible due to the emergence of digital streaming. Because of this, it paved the way to the creation of a huge amount of digital data on the viewers' activity and the movies themselves. One of the most popular use cases of big data on streaming information are recommender systems.</p>
    
<p style="text-align:justify">Recommender systems are helpful in trying to predict the preference or the possible ratings of users for items that they haven't tried yet. They give suggestions based on their algorithm as to what the consumer might prefer to purchase or use next. There are several ways to approach the creation of the recommender system, it could be purely ratings based, popularity based, content based or filtered by user contexts. In this study we tried to implement the content-based and context aware recommendations systems separately and also tried to create a simple combination of the two methods on a long-standing movie rating database Movielens.</p>
    
<p style="text-align:justify">Movielens is a research site with a recommender system powered by online users. Movie recommendations are made using collaborative filtering technology and the ratings that the other members enter. iIt does not provide a streaming platform, but is mainly a recommendation platform for movies and shows. Movielens was created by the group GroupLens Research who specializes in recommender systems, online communities, digital libraries, and GIS. They have several datasets available online. Their datasets, most notably Movielens, have been big contributors in several research endeavors and are widely cited due to its open-source nature.</p>



<b> Problem Statement </b>

<p style="text-align:justify"> With the huge amount of data available on user activity in movie streaming platforms, the team would like to implement several advanced recommender system techniques in order to find out which one works. How might we provide effective movie recommendations using several user specific features (gender, age, occupation) and movie features (title and genre)? </p>

<b>Motivation</b>

<p style="text-align:justify"> This technical report focuses on the feasibility of using several content-based and context aware recommender systems in generating recommendations for this specific movie rating dataset. The insights gathered here can be utilized to effectively create a good combination or pipeline of recommender systems in order to produce the best recommendations. </p>

<h2><font color='darkorange'> II. Methodology </font></h2>


```python
display(_image(filename="method.PNG"))
```


![png](output_12_0.png)


### Data Collection
The team collected the Movielens data and preprocessed it.

### Utility Matrix
The final dataframe was then set up by merging all the preprocessed dataframes into a final table which will be used. It includes the movie id, user id, rating, the genre for content and age, gender and occupation for context. 

### Content-based Recommender System
A content based recommender system was then built using the title and genre as the reference content. 

### Context-aware Recommender System
Three context aware recommender systems were then built which includes context pre-filtering, post-filtering and contextual modeling using different filters such as age gender and occupation. 

### Combined Recommender Systems
Lastly, we assigned weights and combined the results for contextual postfiltering and the content based systems. 

### Result Evaluation
The team then checked the recommendations for subjective evaluation and analysis against the profiles chosen.</p>

<h2><font color='darkorange'> III. Data Processing </font></h2>

### A. Extract and Preprocess Dataset

<p style="text-align:justify">In this technical report, we used Movielens 1M dataset extracted from <a href="https://www.kaggle.com/odedgolden/movielens-1m-dataset">Kaggle</a>. It contains more than a million anonymous ratings from six thousand users who rated at least twenty movies. The database contains four thousand movies from 1995 to 2015. The kaggle dataset is composed of three csv files namely:
    <ul>
        <li><i>movies.csv</i> - It contains the details about the movies like title and genre</li>
        <li><i>users.csv</i> - It contains the details of the users like gender, age, and occupation</li>
        <li><i>ratings.csv</i> - It contains the details about ratings given by the user to a movie</li>
    </ul>
</p>


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 1. Snippet of movies.csv
</b></p>'''))
movies = pd.read_csv('movies.csv')
display(movies.head())
```


<p style="font-size:12px;font-style:default;"><b>
Table 1. Snippet of movies.csv
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieid</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story</td>
      <td>['Animation', "Children's", 'Comedy']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji</td>
      <td>['Adventure', "Children's", 'Fantasy']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men</td>
      <td>['Comedy', 'Romance']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale</td>
      <td>['Comedy', 'Drama']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II</td>
      <td>['Comedy']</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 2. Snippet of users.csv
</b></p>'''))
users = pd.read_csv('users.csv')
display(users.head())
```


<p style="font-size:12px;font-style:default;"><b>
Table 2. Snippet of users.csv
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>gender</th>
      <th>age</th>
      <th>occupations</th>
      <th>zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>M</td>
      <td>56</td>
      <td>16</td>
      <td>70072</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>M</td>
      <td>25</td>
      <td>15</td>
      <td>55117</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>M</td>
      <td>45</td>
      <td>7</td>
      <td>02460</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>M</td>
      <td>25</td>
      <td>20</td>
      <td>55455</td>
    </tr>
  </tbody>
</table>
</div>



```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 3. Snippet of ratings.csv
</b></p>'''))
ratings = pd.read_csv('ratings.csv')
display(ratings.head())
```


<p style="font-size:12px;font-style:default;"><b>
Table 3. Snippet of ratings.csv
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>movieid</th>
      <th>rating</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1193</td>
      <td>5</td>
      <td>978300760</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
      <td>978302109</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
      <td>978301968</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
      <td>978300275</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
      <td>978824291</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">As an initial set of preprocessing the team merged the data into a single dataframe for easier manipulation.</p>


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 4. Snippet of merged dataframes
</b></p>'''))

# merge the user ratings and the movie information
merged = pd.merge(ratings, movies, on='movieid')

# merge it into the user information
final = pd.merge(merged, users, on='userid')

display(final.head())
```


<p style="font-size:12px;font-style:default;"><b>
Table 4. Snippet of merged dataframes
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userid</th>
      <th>movieid</th>
      <th>rating</th>
      <th>timestamp</th>
      <th>title</th>
      <th>genre</th>
      <th>gender</th>
      <th>age</th>
      <th>occupations</th>
      <th>zip</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1193</td>
      <td>5</td>
      <td>978300760</td>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>['Drama']</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>661</td>
      <td>3</td>
      <td>978302109</td>
      <td>James and the Giant Peach</td>
      <td>['Animation', "Children's", 'Musical']</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>914</td>
      <td>3</td>
      <td>978301968</td>
      <td>My Fair Lady</td>
      <td>['Musical', 'Romance']</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3408</td>
      <td>4</td>
      <td>978300275</td>
      <td>Erin Brockovich</td>
      <td>['Drama']</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>2355</td>
      <td>5</td>
      <td>978824291</td>
      <td>Bug's Life, A</td>
      <td>['Animation', "Children's", 'Comedy']</td>
      <td>F</td>
      <td>1</td>
      <td>10</td>
      <td>48067</td>
    </tr>
  </tbody>
</table>
</div>


### B. Data Description

The final data features which were chosen in Movielens data sets can be found in the table below:

<center style="font-size:12px;font-style:default;"><b>Table 5  Table Feature Names, Types and Descriptions</b></center>

| Feature Name | Data Type | Description |
| :- | :- | :- |
| userid | int | Account user identifier |
| movieid | int | Movie identifier code |
| rating | int64 | Ratings given by the user for the particular movie |
| timestamp | datetime | Date the user made the rating |
| title | str | Title of the Movie |
| genre | str | Title of the Movie |
| gender | str | Title of the Movie |
| age | str | Title of the Movie |
| occupations | str | Title of the Movie |
| zip | str | Title of the Movie |

### C. User Profiles

<p style="text-align:justify">We selected two profile from the array of users in the database. First is Maria, a mom and an educator. She is 32 years old. Her movie preferences according to her rating history are more of the comedy and drama type of movies but a lot of the higher rated ones are particularly on children's movies probably due to her kids. Meanwhile another user, Jose, is a 28 years old male graduate student. His movie preferences are more of the classics, and most of them are on the drama genre. Throughout the report, we will reference the recommendations to these two profiles to see if the resulting recommendations are acceptable.</p>


```python
display(_image(filename="profiles.PNG"))
```


![png](output_27_0.png)



```python
# first user's movie preferences

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 6. Maria's Seen & Rated Movies
</b></p>'''))

display(final[final.userid == 3388][['rating', 'title']])
```


<p style="font-size:12px;font-style:default;"><b>
Table 6. Maria's Seen & Rated Movies
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>743997</th>
      <td>4</td>
      <td>Bug's Life, A</td>
    </tr>
    <tr>
      <th>743998</th>
      <td>5</td>
      <td>Princess Bride, The</td>
    </tr>
    <tr>
      <th>743999</th>
      <td>5</td>
      <td>Christmas Story, A</td>
    </tr>
    <tr>
      <th>744000</th>
      <td>4</td>
      <td>Ferris Bueller's Day Off</td>
    </tr>
    <tr>
      <th>744001</th>
      <td>3</td>
      <td>Airplane!</td>
    </tr>
    <tr>
      <th>744002</th>
      <td>4</td>
      <td>Rain Man</td>
    </tr>
    <tr>
      <th>744003</th>
      <td>4</td>
      <td>Stand by Me</td>
    </tr>
    <tr>
      <th>744004</th>
      <td>3</td>
      <td>Strictly Ballroom</td>
    </tr>
    <tr>
      <th>744005</th>
      <td>3</td>
      <td>Raising Arizona</td>
    </tr>
    <tr>
      <th>744006</th>
      <td>3</td>
      <td>Ghostbusters</td>
    </tr>
    <tr>
      <th>744007</th>
      <td>4</td>
      <td>Much Ado About Nothing</td>
    </tr>
    <tr>
      <th>744008</th>
      <td>4</td>
      <td>Babe</td>
    </tr>
    <tr>
      <th>744009</th>
      <td>5</td>
      <td>Hurricane, The</td>
    </tr>
    <tr>
      <th>744010</th>
      <td>5</td>
      <td>Charlotte's Web</td>
    </tr>
    <tr>
      <th>744011</th>
      <td>4</td>
      <td>Jungle Book, The</td>
    </tr>
    <tr>
      <th>744012</th>
      <td>3</td>
      <td>Heart and Souls</td>
    </tr>
    <tr>
      <th>744013</th>
      <td>4</td>
      <td>Patch Adams</td>
    </tr>
    <tr>
      <th>744014</th>
      <td>3</td>
      <td>Lethal Weapon</td>
    </tr>
    <tr>
      <th>744015</th>
      <td>3</td>
      <td>In the Name of the Father</td>
    </tr>
    <tr>
      <th>744016</th>
      <td>1</td>
      <td>Shadow, The</td>
    </tr>
    <tr>
      <th>744017</th>
      <td>3</td>
      <td>Evening Star, The</td>
    </tr>
  </tbody>
</table>
</div>



```python
# second user's movie preferences

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 7. Jose's Seen & Rated Movies
</b></p>'''))

display(final[final.userid == 1406][['rating', 'title']])
```


<p style="font-size:12px;font-style:default;"><b>
Table 7. Jose's Seen & Rated Movies
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rating</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>962332</th>
      <td>5</td>
      <td>Schindler's List</td>
    </tr>
    <tr>
      <th>962333</th>
      <td>3</td>
      <td>Titanic</td>
    </tr>
    <tr>
      <th>962334</th>
      <td>4</td>
      <td>GoodFellas</td>
    </tr>
    <tr>
      <th>962335</th>
      <td>4</td>
      <td>American Beauty</td>
    </tr>
    <tr>
      <th>962336</th>
      <td>4</td>
      <td>Raiders of the Lost Ark</td>
    </tr>
    <tr>
      <th>962337</th>
      <td>4</td>
      <td>Silence of the Lambs, The</td>
    </tr>
    <tr>
      <th>962338</th>
      <td>5</td>
      <td>Shawshank Redemption, The</td>
    </tr>
    <tr>
      <th>962339</th>
      <td>4</td>
      <td>Pulp Fiction</td>
    </tr>
    <tr>
      <th>962340</th>
      <td>4</td>
      <td>Good Will Hunting</td>
    </tr>
    <tr>
      <th>962341</th>
      <td>2</td>
      <td>Fight Club</td>
    </tr>
    <tr>
      <th>962342</th>
      <td>4</td>
      <td>Babe</td>
    </tr>
    <tr>
      <th>962343</th>
      <td>3</td>
      <td>Splash</td>
    </tr>
    <tr>
      <th>962344</th>
      <td>4</td>
      <td>Jerry Maguire</td>
    </tr>
    <tr>
      <th>962345</th>
      <td>4</td>
      <td>Truman Show, The</td>
    </tr>
    <tr>
      <th>962346</th>
      <td>3</td>
      <td>Scent of a Woman</td>
    </tr>
    <tr>
      <th>962347</th>
      <td>5</td>
      <td>Godfather: Part III, The</td>
    </tr>
    <tr>
      <th>962348</th>
      <td>4</td>
      <td>Little Princess, A</td>
    </tr>
    <tr>
      <th>962349</th>
      <td>4</td>
      <td>Me, Myself and Irene</td>
    </tr>
    <tr>
      <th>962350</th>
      <td>4</td>
      <td>Stepmom</td>
    </tr>
    <tr>
      <th>962351</th>
      <td>2</td>
      <td>Robocop 2</td>
    </tr>
  </tbody>
</table>
</div>


<h2><font color='darkorange'> IV. Exploratory Data Analysis </font></h2>

We performed exploratory data analysis particularly to the content and context of our datasets


```python
# renaming list

rename = {"occupations": {0: "other", 1: "academic/educator", 2: "artist", 
                          3: "clerical/admin",  4: "college/grad student",
                          5: "customer service",  6: "doctor/health care",  
                          7: "executive/managerial", 8: "farmer",
                          9: "homemaker", 10: "K-12 student", 11: "lawyer", 
                          12: "programmer", 13: "retired", 
                          14: "sales/marketing", 15: "scientist", 
                          16: "self-employed", 17: "technician/engineer",
                          18: "tradesman/craftsman", 19: "unemployed", 
                          20: "writer"},
         "age": {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 
                 45: "45-49", 50: "50-55", 56: "56+"}}
```

### A.  Top Genre

<p style="text-align:justify">In this section, we explore on the different characteristics of genre which is going to be used as a content for our content-based recommender system. </p>

<b>Count of movies in the Movielens per genre</b>

<p style="text-align:justify">Since most of the movies have multiple genres, we tranform the list of genres into a single column of genres. We then aggregated the genre to determine the count per genre.</p>


```python
df_movies = movies.copy()
display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 1. Top 10 genre with largest movie count.
</b></center>'''))
genre = (df_movies.genre.str.split(',')
                  .apply(lambda x: [re.sub(r'[^\w\s]', '', i) for i in x])
                  .explode()
                  .value_counts()[:10][::-1])

display(HTML(f'''<h3 style="text-align:center">
                Most of the movies in MovieLens have
                <b style="color:{"darkorange"}">
                drama</b> genre
                </h3>'''))

plt.figure(figsize=(10, 6))
genre.plot.barh(color= ['lightgray'] * (len(genre) - 1) + ['darkorange'])
plt.xlabel("Number of Movies", fontsize=15)
plt.ylabel("Genre", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 1. Top 10 genre with largest movie count.
</b></center>



<h3 style="text-align:center">
                Most of the movies in MovieLens have
                <b style="color:darkorange">
                drama</b> genre
                </h3>



![png](output_36_2.png)


<p style="text-align:justify">Drama is the genre that MovieLens invested alot followed by comedy and action.</p>

<b>Count of rated movies in the Movielens per genre</b>

<p style="text-align:justify">Similarly, to get the count of rated movies in the Movielens per genre, we tranform the list of genres into a single column of genres. If a particular movie has multiple genre then all genres under it will be counted for the ratings.</p>


```python
genre = (final.genre.str.split(',')
                  .apply(lambda x: [re.sub(r'[^\w\s]', '', i) for i in x])
                  .explode()
                  .value_counts()[:10][::-1])
display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 2. Top 10 genre with most rated counts.
</b></center>'''))

display(HTML(f'''<h3 style="text-align:center">
                Most rated movies of MovieLens 
                have<b style="color:{"orange"}">
                comedy</b> genre
                </h3>'''))


plt.figure(figsize=(10, 6))
genre.plot.barh(color= ['lightgray'] * (len(genre) - 3) + ['darkorange']
                + ['lightgray'] + ['darkorange'])
plt.xlabel("Number of Rated Movies", fontsize=15)
plt.ylabel("Genre", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 2. Top 10 genre with most rated counts.
</b></center>



<h3 style="text-align:center">
                Most rated movies of MovieLens 
                have<b style="color:orange">
                comedy</b> genre
                </h3>



![png](output_40_2.png)


<p style="text-align:justify">Comedy is the most rated genre in MovieLens. We can also see that Drama is again one of the top rated movies. Having this information we can't tell if these ranking are high or low. We can go deeper by checking the average ratings per genre.</p>

<b>Average ratings per genre</b>

<p style="text-align:justify">The approach to get the average ratings per genre is similar to the previous section, but instead of counting the number of ratings, we get its average.</p>


```python
display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 3. Top 10 genre with highest average ratings.
</b></center>'''))

display(HTML(f'''<h3 style="text-align:center">
                <b style="color:{"orange"}">
                Film Noir</b> has the highest average ratings
                </h3>'''))


plt.figure(figsize=(10, 6))

eda = final.copy()
eda['genre'] = eda.genre.apply(eval)
(eda.loc[:, ['genre', 'rating']].explode('genre')
   .groupby('genre')['rating'].mean()
   .sort_values(ascending=False)[:10][::-1]
   .plot.barh(color= ['lightgray'] * (len(genre) - 4) + ['darkorange']
                     + ['lightgray'] * 2 + ['darkorange']))
plt.xlabel("Ratings", fontsize=15)
plt.ylabel("Genre", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


```


<center style="font-size:12px;font-style:default;"><b>
Figure 3. Top 10 genre with highest average ratings.
</b></center>



<h3 style="text-align:center">
                <b style="color:orange">
                Film Noir</b> has the highest average ratings
                </h3>



![png](output_44_2.png)


<p style="text-align:justify">Film Noir or Hollywood Crime dramas has the highest average ratings among the genre of MovieLens. Overall, drama seems to be on top in terms of number of movies, number of rated movies, and average rating. </p>


### B. Number of Users per Category

<p style="text-align:justify"> We also tried to see which group per category has the most number of users. Gender-wise, we saw that there are more male users in the platform. Meawnhile, for the age brackets, 25-34 year olds have the most users followed by the 35-44 year old bracket. Last, for the occupation, most of them are also college/graduate students. followed by executive and academics.</p>


```python
# plot user by gender

user_plot = users.replace(rename)

display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 4. User Distribution per Gender.
</b></center>'''))

gender_plot = (user_plot.gender.value_counts(ascending=True))

display(HTML(f'''<h3 style="text-align:center">
                Number of users based on<b style="color:{"orange"}">
                gender</b>
                </h3>'''))


plt.figure(figsize=(10, 6))
gender_plot.plot.barh(color= ['darkgray'] * (len(gender_plot) - 1) + ['darkorange'])
plt.xlabel("Number of Users", fontsize=15)
plt.ylabel("Gender", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 4. User Distribution per Gender.
</b></center>



<h3 style="text-align:center">
                Number of users based on<b style="color:orange">
                gender</b>
                </h3>



![png](output_48_2.png)



```python
# plot user by age

age_plot = (user_plot.age.value_counts(ascending=True))

display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 5. User Distribution per Age Bracket.
</b></center>'''))

display(HTML(f'''<h3 style="text-align:center">
                Number of users based on<b style="color:{"orange"}">
                age bracket</b>
                </h3>'''))


plt.figure(figsize=(10, 6))
age_plot.plot.barh(color= ['darkgray'] * (len(age_plot) - 1) + ['darkorange'])
plt.xlabel("Number of Users", fontsize=15)
plt.ylabel("Age Bracket", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 5. User Distribution per Age Bracket.
</b></center>



<h3 style="text-align:center">
                Number of users based on<b style="color:orange">
                age bracket</b>
                </h3>



![png](output_49_2.png)



```python
# plot user by occupation

occ_plot = (user_plot.occupations.value_counts(ascending=True))

display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 6. User Distribution per Occupation.
</b></center>'''))

display(HTML(f'''<h3 style="text-align:center">
                Number of users based on<b style="color:{"orange"}">
                occupation</b>
                </h3>'''))


plt.figure(figsize=(10, 6))
occ_plot.plot.barh(color= ['darkgray'] * (len(occ_plot) - 1) + ['darkorange'])
plt.xlabel("Number of Users", fontsize=15)
plt.ylabel("Occupation", fontsize=15)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 6. User Distribution per Occupation.
</b></center>



<h3 style="text-align:center">
                Number of users based on<b style="color:orange">
                occupation</b>
                </h3>



![png](output_50_2.png)


### C. Average Ratings per Category

<p style="text-align:justify">Aside from getting the number of users per category we might want to get the average ratings per category as well. We might explore on the behavior of the different groups in our database.</p>


```python
fig, ax = plt.subplots(1, 1, figsize=(8,6))

display(HTML('''<center style="font-size:12px;font-style:default;"><b>
Figure 7. Average Ratings per Category.
</b></center>'''))

display(HTML(f'''<h3 style="text-align:center">
                <b style="color:{"darkorange"}">
                Female</b> and <b style="color:{"darkorange"}">Older</b>
                generation tend to rate higher
                </h3>'''))

(final.groupby('gender')['rating'].mean()
 .sort_values()
 .plot.barh(color=['lightgray','darkorange'], xlim=(3,4), ax=ax))

ax.set_xlabel("Rating", fontsize=15)
ax.set_ylabel("Gender", fontsize=15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


fig, ax = plt.subplots(1, 2, figsize=(25,8))

(final.groupby('age')['rating'].mean()
 .sort_values().plot.barh(ax=ax[0], color= ['lightgray'] * 
                          (len(genre) - 4) + ['darkorange']
                          + ['lightgray'] * 2 + ['orange'], xlim=(3,4)))
ax[0].set_xlabel("Rating", fontsize=15)
ax[0].set_ylabel("Age group", fontsize=15)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)

df = final.copy()
df['occupations'] = df.occupations.apply(lambda x: rename['occupations'][x])
(df.groupby('occupations')['rating'].mean()
 .sort_values(ascending=False)[:10][::-1]
 .plot.barh(ax=ax[1],color=['lightgray']  * 9 + ['darkorange'], xlim=(3,4)))
ax[1].set_xlabel("Rating", fontsize=15)
ax[1].set_ylabel("Age group", fontsize=15)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)

fig.tight_layout()
```


<center style="font-size:12px;font-style:default;"><b>
Figure 7. Average Ratings per Category.
</b></center>



<h3 style="text-align:center">
                <b style="color:darkorange">
                Female</b> and <b style="color:darkorange">Older</b>
                generation tend to rate higher
                </h3>



![png](output_53_2.png)



![png](output_53_3.png)


<p style="text-align:justify">It seems that the rating increases as the age group gets older, it can also be seen by the occupation who rated it higher are retired users. It looks like the movies in movielens are more appreciated by older generation. Interestingly, female also tend to give a little higher ratings than male.</p>

<h2><font color='darkorange'> V. Results and Discussion </font></h2>

### A. Content-based Recommender System

<p style="text-align:justify">Content-based recommender system uses content metadata and user-profiles. By using content-based recommender system, we solve the problem of cold start that are present in collaborative filtering, it also recommend without regards to the popularity of the movie, and lastly it is also interpretable and intuitive. In this technical report, we used title and genre as our content metadata.$^2$</p>

<ol>
    <li> We used <code>nltk punkt</code> to properly tokenize the words in English that are going to be useful in the succeeding preprocessing of reviews.
    <li> We then performed casefolding to be able to have a case-insensitive reviews such that we want <code>GOOD</code> and <code>good</code> to be counted as single word. 
    <li> We also decided to perform lemmatization from <code>nltk WordNetLemmatizer</code> to the tokens such that we want <code>bed</code> and <code>beds</code> to be counted as single word.
    <li> We removed english stopwords from <code>nltk stopwords</code> to prevent low-level information such as <code>about</code>, <code>without</code>, etc from our movie titles in order to give more importance to the relevant information
<li> Lastly, we filtered words with less than 3 character length prevent words like <code>the</code>, <code>be</code> that wouldn't have importance for our analysis 
</ol></p>


```python
df = pd.read_csv('final.csv')
df_movies = pd.read_csv('movies.csv')

def preprocess_movies(df_movies):
    """ Preprocess the content metadata of the movie
    
    Parameters
    ===========
    df_movies    :    pandas.DataFrame
                      database
    
    Returns
    ===========
    preprocess_movies   :  tuple
                           tuple of database and tfidf transformed data
    """
    df_movies['title_genre'] = df_movies.title +  ' '  + df_movies.genre
    df_movies['title_genre'] = df_movies.title_genre.str.replace(r'[^\w\s]', 
                                                            '', regex=True)

    # tokenize
    tokenize = df_movies.title_genre.apply(nltk.word_tokenize)

    # casefold
    lower_case = tokenize.apply(lambda x:list(map(lambda y: y.casefold(), x)))


    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemmatize = lower_case.apply(lambda x: list(map(lemmatizer.lemmatize,
                                                             x)))

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_stopwords = lemmatize.apply(lambda x:
                                        list(filter(lambda y: y
                                                not in stop_words,
                                                              x)))

    # filter words with less than 3 character length
    filtered_words = filtered_stopwords.apply(lambda x:
                                                       list(filter(lambda y:
                                                                   len(y) > 3,
                                                                   x)))

    df_movies['clean'] = (filtered_words.apply(' '.join))

    tfidf_vectorizer = TfidfVectorizer(token_pattern=r'\b[a-z]+\b', 
                                       ngram_range=(1, 2),
                                       max_df=0.8,
                                       min_df=0.01)


    tfidf = tfidf_vectorizer.fit_transform(df_movies.clean)

    svd = TruncatedSVD(n_components=20, random_state=0)
    X_new = svd.fit_transform(tfidf)
    
    return df_movies, X_new

```

<p style="text-align:justify">Now we have to convert are text data into a machine readable format which is a number. We can do this by training a Tf-idf vector using <code>sklearn.feature_extraction.text.TfidfVectorizer</code>. We have used the following parameters for the <code>TfidfVectorizer</code>:
<ul>
    <li>token_pattern : <code>r'\b[a-z]+\b'</code> - We have defined words as strictly composed of letters in the alphabet prefixed and suffixed by a word boundary</li>
    <li>ngram_range : <code>(1, 2)</code> - We used 1 to 2 ngrams to capture single and compound words such as <i>white house</i>, and <i>house</i></li>
    <li>max_df : <code>0.8</code> - We set the limit of the maximum document frequency to be 80% to remove words that appear too frequently</li>
    <li>min_df : <code>0.01</code> - We set the limit of the minimum document frequency to be 1% to ignore the words that appear too infrequently to our data that would improve also the performance of our model.</li></ul></p>


```python
def content_recom(user, df, df_movies, X_new, top_k=10, **kwargs):
    """ Perform content model based recommender sytem for a single user
    
    Parameters
    ===========
    user        :    int 
                     user id
    df          :    pandas.DataFrame
                     full database
    df_movies   :    pandas.DataFrame
                     movies database
    X_new       :    numpy.ndarray
                     tfidf transformed features
    top_k       :    int
                     number of recommendations
    
    Returns
    ===========
    content_recom  : pandas.DataFrame
                     dataframe of recommended movies
    """
    if 'pre' in kwargs.keys():
        df = df[df[kwargs['context_var']] == kwargs['context_var_value']]
        df = df.reset_index()
    
    recom = {}
    df_user = df[df.userid==user].copy().sort_values(by='movieid') 
    profile = {
    'user': df_user.userid.unique()[0],
    'gender' : df_user.gender.unique()[0],
    'timestamp' : None,
    'age' : df_user.age.unique()[0],
    'occupations' : df_user.occupations.unique()[0],
    'zip' : df_user.zip.unique()[0],
    }
    
    df_out = pd.DataFrame()
    
    seen = set(df_user.movieid)
    unseen = set(df.movieid) - seen
    df_movies = df_movies.sort_values(by='movieid')
    movie_index_seen = df_movies[df_movies.movieid.isin(
            sorted(seen))].index

    X = X_new[movie_index_seen]
    y = df_user.rating.to_numpy()
    movie_index_unseen = df_movies[df_movies.movieid.isin(
        sorted(unseen))].index
    X_test = X_new[movie_index_unseen, :]
    model = Ridge(random_state=143)

    try:
        model.fit(X, y)
        new_rating = model.predict(X_test)
        limit = len(new_rating) if 'post' in kwargs.keys() else top_k
        top_recom = np.argsort(new_rating, kind='mergesort')[::-1][:limit]
        top_item = np.array(sorted(unseen))[top_recom]
        for i, item in enumerate(top_item):
            temp = profile.copy()
            temp['movieid'] = item
            temp['title'] = df_movies[df_movies.movieid == 
                                      item].title.to_list()[0]
            temp['genre'] = df_movies[df_movies.movieid == 
                                      item].genre.to_list()[0]
            temp['rating'] = new_rating[top_recom[i]]
            df_out = df_out.append(pd.DataFrame(temp, 
                                   index=[0])).reset_index(drop=True)
        
    except ValueError:
        pass
    
    if 'post' in kwargs.keys():
        df_out = df_out[df_out[kwargs['context_var']] == 
                        kwargs['context_var_value']]
    
    return df_out[:top_k]
```

<p style="text-align:justify">Let us look on how our model performed on our two customer persona:</p>

#### Maria's Recommendations


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 8. Maria's Recommended movies based on content metadata
</b></p>'''))

df_movies, X_new = preprocess_movies(df_movies)
display(content_recom(3388, df, df_movies, X_new).iloc[:, [0, 6, 7, 8]])
```


<p style="font-size:12px;font-style:default;"><b>
Table 8. Maria's Recommended movies based on content metadata
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>movieid</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3388</td>
      <td>3945</td>
      <td>Digimon: The Movie</td>
      <td>['Adventure', 'Animation', "Children's"]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3388</td>
      <td>2899</td>
      <td>Gulliver's Travels</td>
      <td>['Adventure', 'Animation', "Children's"]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3388</td>
      <td>2800</td>
      <td>Little Nemo: Adventures in Slumberland</td>
      <td>['Animation', "Children's"]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3388</td>
      <td>2116</td>
      <td>Lord of the Rings, The</td>
      <td>['Adventure', 'Animation', "Children's", 'Sci-...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3388</td>
      <td>2099</td>
      <td>Song of the South</td>
      <td>['Adventure', 'Animation', "Children's", 'Musi...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3388</td>
      <td>1030</td>
      <td>Pete's Dragon</td>
      <td>['Adventure', 'Animation', "Children's", 'Musi...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3388</td>
      <td>1881</td>
      <td>Quest for Camelot</td>
      <td>['Adventure', 'Animation', "Children's", 'Fant...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3388</td>
      <td>3754</td>
      <td>Adventures of Rocky and Bullwinkle, The</td>
      <td>['Animation', "Children's", 'Comedy']</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3388</td>
      <td>3799</td>
      <td>Pokémon the Movie 2000</td>
      <td>['Animation', "Children's"]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3388</td>
      <td>3615</td>
      <td>Dinosaur</td>
      <td>['Animation', "Children's"]</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">The recommended movies for Maria revolves around the genre of adventure, animation, and children which is similar to her profile that we presented earlier which means we got a good result of recommendation based on the genre. These movies are digimon, little nemo, the lord of the rings, song of the south, and many more.</p>

#### Jose's Recommendations


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 9. Jose's Recommended movies based on content metadata
</b></p>'''))

df_movies, X_new = preprocess_movies(df_movies)
display(content_recom(1406, df, df_movies, X_new).iloc[:, [0, 6, 7, 8]])
```


<p style="font-size:12px;font-style:default;"><b>
Table 9. Jose's Recommended movies based on content metadata
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>movieid</th>
      <th>title</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1406</td>
      <td>2764</td>
      <td>Thomas Crown Affair, The</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1406</td>
      <td>1620</td>
      <td>Kiss the Girls</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1406</td>
      <td>1598</td>
      <td>Desperate Measures</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1406</td>
      <td>1227</td>
      <td>Once Upon a Time in America</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1406</td>
      <td>608</td>
      <td>Fargo</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1406</td>
      <td>463</td>
      <td>Guilty as Sin</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1406</td>
      <td>259</td>
      <td>Kiss of Death</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1406</td>
      <td>149</td>
      <td>Amateur</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1406</td>
      <td>22</td>
      <td>Copycat</td>
      <td>['Crime', 'Drama', 'Thriller']</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1406</td>
      <td>3900</td>
      <td>Crime and Punishment in Suburbia</td>
      <td>['Comedy', 'Drama']</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">Meanwhile for Jose, the recommended movies for him with predicted rating of almost 4, are thomas crown affair, kiss the girls, desparate measures, once upon a time in america. Personally I am not familiar with these movies but they are all under crime, drama, and thriller which is again similar to the profile that we have presented earlier.</p>

### B. Context-aware Recommender System


```python
# prepare matrix for context-aware RS

sample = final.copy()
sample = sample[['userid', 'movieid', 'rating', 
                 'gender', 'age', 'occupations']]
encode_nums = {"gender":     {"M": 0, "F": 1},
               "age": {1 : 0, 18 : 1, 25 : 2, 35 : 3, 
                       45 : 4, 50 : 5, 56 : 6 }}
sample = sample.replace(encode_nums)
```

<p style="text-align:justify">For context aware systems, we tried three models, context pre-filtering, post-filtering and contextual modeling. For the pre and post filtering methods we used knn with k value of 5. Meanwhile for contextual modeling we used a latent factor model and Factorization Machines Regressor learning algorithm.$^3$</p>

<p style="text-align:justify">In general one thing that we noticed with context-aware recommendations are that they are heavily influenced by the filters used more than the viewing history. Evidenced by the suggestions to Maria in the reulsts below which are not exactly child friendly and are more adult appropriate. There could be a problem if the user profile doesnt match the activity.</p>

#### Context Pre-filtering

<p style="text-align:justify">For Maria the mom, we can see that there are some common suggestions when filtered only by gender or by age. No results were generated when we used the occupation filter due to the effect of pre-filtering the initial utility matrix which made it harder to find similar items for the recommender system. That is one of the drawbacks of pre-filtering. The same reason caused the model to fail when we tried to implement 2 or 3 filters together.</p>


```python
# recommendations with context filters

cpre = ContextPreFiltering(sample, ['gender'], (1, 5))
cpre.fit([1])
cpre_recommendations1 = cpre.show_top_k(3388)
recom_table = final.loc[[i[0] for i in cpre_recommendations1], ['title']]
recom_table['ratings'] = [i[1] for i in cpre_recommendations1]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 10. Maria's Recommendations - Gender Filter (Female)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 10. Maria's Recommendations - Gender Filter (Female)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1450</th>
      <td>Total Recall</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3292</th>
      <td>Rocky Horror Picture Show, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2538</th>
      <td>Meet the Parents</td>
      <td>5</td>
    </tr>
    <tr>
      <th>687</th>
      <td>Dragonheart</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1780</th>
      <td>Spellbound</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2304</th>
      <td>Shawshank Redemption, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>681</th>
      <td>Lethal Weapon 3</td>
      <td>5</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3817</th>
      <td>Desperado</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>5</td>
    </tr>
    <tr>
      <th>394</th>
      <td>Story of Us, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2332</th>
      <td>Big</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>Sneakers</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1071</th>
      <td>Army of Darkness</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>854</th>
      <td>Parenthood</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3012</th>
      <td>Manchurian Candidate, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3607</th>
      <td>Candyman</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2760</th>
      <td>Out-of-Towners, The</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpre = ContextPreFiltering(sample, ['age'], (1, 5))
cpre.fit([2])
cpre_recommendations1 = cpre.show_top_k(3388)
recom_table = final.loc[[i[0] for i in cpre_recommendations1], ['title']]
recom_table['ratings'] = [i[1] for i in cpre_recommendations1]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 11. Maria's Recommendations - Age Filter (25-34 years old)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 11. Maria's Recommendations - Age Filter (25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1300</th>
      <td>Sixteen Candles</td>
      <td>5</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Babe: Pig in the City</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>Howling, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Independence Day (ID4)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>853</th>
      <td>Highlander</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1780</th>
      <td>Spellbound</td>
      <td>5</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2358</th>
      <td>Few Good Men, A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>Milk Money</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>Twelve Monkeys</td>
      <td>5</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Newton Boys, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>578</th>
      <td>Hidden, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Mr. Saturday Night</td>
      <td>5</td>
    </tr>
    <tr>
      <th>696</th>
      <td>Indiana Jones and the Temple of Doom</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2309</th>
      <td>Romancing the Stone</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2466</th>
      <td>Heat</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>Saving Private Ryan</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


Luckily for Jose, all of the single filters worked but when we tried to use double and triple filters it did not work just like for user Maria.


```python
# recommendations with context filters

cpre = ContextPreFiltering(sample, ['gender'], (1, 5))
cpre.fit([0])
cpre_recommendations1 = cpre.show_top_k(1406)
recom_table = final.loc[[i[0] for i in cpre_recommendations1], ['title']]
recom_table['ratings'] = [i[1] for i in cpre_recommendations1]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 12. Jose's Recommendations - Gender Filter (Male)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 12. Jose's Recommendations - Gender Filter (Male)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3233</th>
      <td>Private Parts</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3517</th>
      <td>Kids</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3656</th>
      <td>Junk Mail</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>439</th>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>985</th>
      <td>Indiana Jones and the Temple of Doom</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Dances with Wolves</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>989</th>
      <td>King Kong</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>Go</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3280</th>
      <td>Teenage Mutant Ninja Turtles</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1046</th>
      <td>Carrie</td>
      <td>4.990800</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>Toy Story 2</td>
      <td>4.896184</td>
    </tr>
    <tr>
      <th>2360</th>
      <td>Total Recall</td>
      <td>4.838214</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>Being There</td>
      <td>4.788959</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.764178</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.760959</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Heavy Metal</td>
      <td>4.712866</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>4.697141</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpre = ContextPreFiltering(sample, ['age'], (1, 5))
cpre.fit([2])
cpre_recommendations1 = cpre.show_top_k(1406)
recom_table = final.loc[[i[0] for i in cpre_recommendations1], ['title']]
recom_table['ratings'] = [i[1] for i in cpre_recommendations1]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 13. Jose's Recommendations - Age Filter (25-34 years old)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 13. Jose's Recommendations - Age Filter (25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3022</th>
      <td>Clockers</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1795</th>
      <td>Howling, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3140</th>
      <td>Good, The Bad and The Ugly, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Independence Day (ID4)</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>853</th>
      <td>Highlander</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1780</th>
      <td>Spellbound</td>
      <td>5</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2358</th>
      <td>Few Good Men, A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>Milk Money</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1553</th>
      <td>Twelve Monkeys</td>
      <td>5</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Newton Boys, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>Mr. Saturday Night</td>
      <td>5</td>
    </tr>
    <tr>
      <th>696</th>
      <td>Indiana Jones and the Temple of Doom</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2309</th>
      <td>Romancing the Stone</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2466</th>
      <td>Heat</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>Saving Private Ryan</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpre = ContextPreFiltering(sample, ['occupations'], (1, 5))
cpre.fit([4])
cpre_recommendations1 = cpre.show_top_k(1406)
recom_table = final.loc[[i[0] for i in cpre_recommendations1], ['title']]
recom_table['ratings'] = [i[1] for i in cpre_recommendations1]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 14. Jose's Recommendations - Occupation Filter (Grad Student)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 14. Jose's Recommendations - Occupation Filter (Grad Student)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>326</th>
      <td>Blast from the Past</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3637</th>
      <td>What Planet Are You From?</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3073</th>
      <td>League of Their Own, A</td>
      <td>5</td>
    </tr>
    <tr>
      <th>583</th>
      <td>Laura</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3711</th>
      <td>Star Wars: Episode IV - A New Hope</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2131</th>
      <td>Dances with Wolves</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2342</th>
      <td>Mulan</td>
      <td>5</td>
    </tr>
    <tr>
      <th>602</th>
      <td>Fly, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>669</th>
      <td>Star Wars: Episode V - The Empire Strikes Back</td>
      <td>5</td>
    </tr>
    <tr>
      <th>621</th>
      <td>Wizard of Oz, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2823</th>
      <td>Milk Money</td>
      <td>5</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1420</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2981</th>
      <td>Chicken Run</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1749</th>
      <td>Mummy's Hand, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2675</th>
      <td>Waterboy, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2215</th>
      <td>Rear Window</td>
      <td>5</td>
    </tr>
    <tr>
      <th>462</th>
      <td>X-Files: Fight the Future, The</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1117</th>
      <td>101 Dalmatians</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>


#### Context Post-filtering

<p style="text-align:justify">Next for context post-filtering, we do the recommendation system first before we actually filter them using our context filters. There was no problem creating the recommender system first because the uitility matrix used was still complete. This worked better for both users since we used the complete original matrix to create the recommendations.</p>

<p style="text-align:justify">Compared to pre-filtering we have more overlaps across the models. It also worked for two and three filters at the same time, both for Maria and Jose.</p>


```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['gender'], (1, 5))
cpost.fit([1])
cpost_recommendations1 = cpost.show_top_k(3388)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 15. Maria's Recommendations - Gender Filter (Female)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 15. Maria's Recommendations - Gender Filter (Female)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3881</th>
      <td>Mosquito Coast, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3607</th>
      <td>Candyman</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3382</th>
      <td>Sneakers</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>4.965224</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>Tales from the Crypt Presents: Bordello of Blood</td>
      <td>4.911797</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>4.823853</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Lord of the Flies</td>
      <td>4.797265</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>When Harry Met Sally...</td>
      <td>4.776346</td>
    </tr>
    <tr>
      <th>2324</th>
      <td>Christmas Story, A</td>
      <td>4.774076</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Babe: Pig in the City</td>
      <td>4.712796</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.702023</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.699886</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman Returns</td>
      <td>4.697960</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Whole Nine Yards, The</td>
      <td>4.679853</td>
    </tr>
    <tr>
      <th>898</th>
      <td>Aladdin and the King of Thieves</td>
      <td>4.653764</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Scent of a Woman</td>
      <td>4.650804</td>
    </tr>
    <tr>
      <th>1131</th>
      <td>Mummy's Tomb, The</td>
      <td>4.626509</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>4.591538</td>
    </tr>
    <tr>
      <th>969</th>
      <td>Conspiracy Theory</td>
      <td>4.586533</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['age'], (1, 5))
cpost.fit([2])
cpost_recommendations1 = cpost.show_top_k(3388)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 16. Maria's Recommendations - Age Filter (25-34 years old)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 16. Maria's Recommendations - Age Filter (25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>578</th>
      <td>Hidden, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>Ref, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>Tales from the Crypt Presents: Bordello of Blood</td>
      <td>4.883656</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>4.823853</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman Returns</td>
      <td>4.814538</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>When Harry Met Sally...</td>
      <td>4.800956</td>
    </tr>
    <tr>
      <th>2324</th>
      <td>Christmas Story, A</td>
      <td>4.765292</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.757569</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.702023</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Whole Nine Yards, The</td>
      <td>4.674454</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.666459</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Scent of a Woman</td>
      <td>4.634485</td>
    </tr>
    <tr>
      <th>2940</th>
      <td>American Beauty</td>
      <td>4.601062</td>
    </tr>
    <tr>
      <th>1131</th>
      <td>Mummy's Tomb, The</td>
      <td>4.600170</td>
    </tr>
    <tr>
      <th>811</th>
      <td>GoldenEye</td>
      <td>4.596186</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>4.591538</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Lord of the Flies</td>
      <td>4.588688</td>
    </tr>
    <tr>
      <th>898</th>
      <td>Aladdin and the King of Thieves</td>
      <td>4.584097</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['occupations'], (1, 5))
cpost.fit([1])
cpost_recommendations1 = cpost.show_top_k(3388)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 17. Maria's Recommendations - Occupation Filter (Academic)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 17. Maria's Recommendations - Occupation Filter (Academic)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3233</th>
      <td>Private Parts</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3881</th>
      <td>Mosquito Coast, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>Ref, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>4.965224</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>Tales from the Crypt Presents: Bordello of Blood</td>
      <td>4.883656</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Lord of the Flies</td>
      <td>4.797265</td>
    </tr>
    <tr>
      <th>2324</th>
      <td>Christmas Story, A</td>
      <td>4.774667</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.757569</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>When Harry Met Sally...</td>
      <td>4.729246</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.702023</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>Scream</td>
      <td>4.680457</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Scent of a Woman</td>
      <td>4.650804</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Whole Nine Yards, The</td>
      <td>4.622984</td>
    </tr>
    <tr>
      <th>3359</th>
      <td>Mommie Dearest</td>
      <td>4.614696</td>
    </tr>
    <tr>
      <th>913</th>
      <td>Running Man, The</td>
      <td>4.612756</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.607769</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman Returns</td>
      <td>4.597258</td>
    </tr>
    <tr>
      <th>811</th>
      <td>GoldenEye</td>
      <td>4.596186</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>4.591538</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['occupations'], (1, 5))
cpost.fit([1])
cpost_recommendations1 = cpost.show_top_k(3388)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 18. Maria's Recommendations - Gender & Age Filter 
(Female & 25-34 years old)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 18. Maria's Recommendations - Gender & Age Filter 
(Female & 25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3233</th>
      <td>Private Parts</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3881</th>
      <td>Mosquito Coast, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>2905</th>
      <td>Ref, The</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>4.965224</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>Tales from the Crypt Presents: Bordello of Blood</td>
      <td>4.883656</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Lord of the Flies</td>
      <td>4.797265</td>
    </tr>
    <tr>
      <th>2324</th>
      <td>Christmas Story, A</td>
      <td>4.774667</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.757569</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>When Harry Met Sally...</td>
      <td>4.729246</td>
    </tr>
    <tr>
      <th>53</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.702023</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>Scream</td>
      <td>4.680457</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Scent of a Woman</td>
      <td>4.650804</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Whole Nine Yards, The</td>
      <td>4.622984</td>
    </tr>
    <tr>
      <th>3359</th>
      <td>Mommie Dearest</td>
      <td>4.614696</td>
    </tr>
    <tr>
      <th>913</th>
      <td>Running Man, The</td>
      <td>4.612756</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.607769</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman Returns</td>
      <td>4.597258</td>
    </tr>
    <tr>
      <th>811</th>
      <td>GoldenEye</td>
      <td>4.596186</td>
    </tr>
    <tr>
      <th>3245</th>
      <td>Chasing Amy</td>
      <td>4.591538</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['gender', 'age', 'occupations'], (1, 5))
cpost.fit([1, 2, 1])
cpost_recommendations1 = cpost.show_top_k(3388)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 19. Maria's Recommendations - Gender, Age & Occupation Filter 
(Female, 25-34 years old, Academe)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 19. Maria's Recommendations - Gender, Age & Occupation Filter 
(Female, 25-34 years old, Academe)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2324</th>
      <td>Christmas Story, A</td>
      <td>4.991697</td>
    </tr>
    <tr>
      <th>3134</th>
      <td>Alien</td>
      <td>4.965224</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>Tales from the Crypt Presents: Bordello of Blood</td>
      <td>4.953422</td>
    </tr>
    <tr>
      <th>1250</th>
      <td>When Harry Met Sally...</td>
      <td>4.880179</td>
    </tr>
    <tr>
      <th>1260</th>
      <td>Batman Returns</td>
      <td>4.859958</td>
    </tr>
    <tr>
      <th>363</th>
      <td>Lord of the Flies</td>
      <td>4.797265</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.735763</td>
    </tr>
    <tr>
      <th>720</th>
      <td>Babe: Pig in the City</td>
      <td>4.712796</td>
    </tr>
    <tr>
      <th>1132</th>
      <td>Scream</td>
      <td>4.680457</td>
    </tr>
    <tr>
      <th>1300</th>
      <td>Sixteen Candles</td>
      <td>4.671334</td>
    </tr>
    <tr>
      <th>745</th>
      <td>Scent of a Woman</td>
      <td>4.650804</td>
    </tr>
    <tr>
      <th>914</th>
      <td>Last Action Hero</td>
      <td>4.639556</td>
    </tr>
    <tr>
      <th>932</th>
      <td>Back to the Future</td>
      <td>4.624600</td>
    </tr>
    <tr>
      <th>1272</th>
      <td>Blues Brothers, The</td>
      <td>4.619182</td>
    </tr>
    <tr>
      <th>3359</th>
      <td>Mommie Dearest</td>
      <td>4.614696</td>
    </tr>
    <tr>
      <th>913</th>
      <td>Running Man, The</td>
      <td>4.612756</td>
    </tr>
    <tr>
      <th>2940</th>
      <td>American Beauty</td>
      <td>4.601062</td>
    </tr>
    <tr>
      <th>1</th>
      <td>James and the Giant Peach</td>
      <td>4.568617</td>
    </tr>
    <tr>
      <th>1545</th>
      <td>Jacob's Ladder</td>
      <td>4.524276</td>
    </tr>
    <tr>
      <th>953</th>
      <td>Braveheart</td>
      <td>4.515461</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['gender'], (1, 5))
cpost.fit([0])
cpost_recommendations1 = cpost.show_top_k(1406)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 20. Jose's Recommendations - Gender Filter (Male)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 20. Jose's Recommendations - Gender Filter (Male)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>Go</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>989</th>
      <td>King Kong</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3656</th>
      <td>Junk Mail</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3233</th>
      <td>Private Parts</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3280</th>
      <td>Teenage Mutant Ninja Turtles</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3888</th>
      <td>Three Kings</td>
      <td>4.989583</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.881376</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>Blazing Saddles</td>
      <td>4.815147</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Payback</td>
      <td>4.745304</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>Toy Story 2</td>
      <td>4.665717</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Super Mario Bros.</td>
      <td>4.650754</td>
    </tr>
    <tr>
      <th>3517</th>
      <td>Kids</td>
      <td>4.635872</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>4.628656</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.620448</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens)</td>
      <td>4.614540</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.600978</td>
    </tr>
    <tr>
      <th>930</th>
      <td>Pleasantville</td>
      <td>4.594709</td>
    </tr>
    <tr>
      <th>904</th>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
      <td>4.591398</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['age'], (1, 5))
cpost.fit([2])
cpost_recommendations1 = cpost.show_top_k(1406)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 21. Jose's Recommendations - Age Filter (25-34 years old)
</b></p>'''))

display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 21. Jose's Recommendations - Age Filter (25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.881376</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>Blazing Saddles</td>
      <td>4.815147</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Payback</td>
      <td>4.799873</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Super Mario Bros.</td>
      <td>4.700580</td>
    </tr>
    <tr>
      <th>2708</th>
      <td>Flubber</td>
      <td>4.682580</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>Toy Story 2</td>
      <td>4.665717</td>
    </tr>
    <tr>
      <th>3517</th>
      <td>Kids</td>
      <td>4.635872</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>4.628656</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.625469</td>
    </tr>
    <tr>
      <th>922</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.623585</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens)</td>
      <td>4.614540</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.606173</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Thelma &amp; Louise</td>
      <td>4.602562</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>Witness</td>
      <td>4.601180</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>Queen Margot (La Reine Margot)</td>
      <td>4.599793</td>
    </tr>
    <tr>
      <th>904</th>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
      <td>4.596262</td>
    </tr>
    <tr>
      <th>928</th>
      <td>Ferris Bueller's Day Off</td>
      <td>4.592893</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.592397</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['occupations'], (1, 5))
cpost.fit([4])
cpost_recommendations1 = cpost.show_top_k(1406)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 22. Jose's Recommendations - Occupation Filter (Grad Student)
</b></p>'''))
display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 22. Jose's Recommendations - Occupation Filter (Grad Student)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.881376</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Super Mario Bros.</td>
      <td>4.794592</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Payback</td>
      <td>4.747700</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Blast from the Past</td>
      <td>4.671838</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>Hercules</td>
      <td>4.634679</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>Queen Margot (La Reine Margot)</td>
      <td>4.629279</td>
    </tr>
    <tr>
      <th>922</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.623585</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens)</td>
      <td>4.614540</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>Witness</td>
      <td>4.601180</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>Forbidden Planet</td>
      <td>4.600113</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>Scream 3</td>
      <td>4.594842</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.594663</td>
    </tr>
    <tr>
      <th>930</th>
      <td>Pleasantville</td>
      <td>4.551803</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.544423</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>Raising Arizona</td>
      <td>4.542249</td>
    </tr>
    <tr>
      <th>2357</th>
      <td>Mission: Impossible</td>
      <td>4.538931</td>
    </tr>
    <tr>
      <th>904</th>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
      <td>4.538436</td>
    </tr>
    <tr>
      <th>903</th>
      <td>Home Alone 2: Lost in New York</td>
      <td>4.533733</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Thelma &amp; Louise</td>
      <td>4.532401</td>
    </tr>
    <tr>
      <th>950</th>
      <td>Maverick</td>
      <td>4.508458</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['gender', 'age'], (1, 5))
cpost.fit([0, 2])
cpost_recommendations1 = cpost.show_top_k(1406)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 23. Jose's Recommendations - Gender & Age Filter (Male, 25-34 years old)
</b></p>'''))

display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 23. Jose's Recommendations - Gender & Age Filter (Male, 25-34 years old)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>787</th>
      <td>Lethal Weapon</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3172</th>
      <td>Pi</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.881376</td>
    </tr>
    <tr>
      <th>2444</th>
      <td>Blazing Saddles</td>
      <td>4.815147</td>
    </tr>
    <tr>
      <th>954</th>
      <td>Payback</td>
      <td>4.772364</td>
    </tr>
    <tr>
      <th>2708</th>
      <td>Flubber</td>
      <td>4.682580</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>Toy Story 2</td>
      <td>4.665717</td>
    </tr>
    <tr>
      <th>3410</th>
      <td>Omega Man, The</td>
      <td>4.628656</td>
    </tr>
    <tr>
      <th>922</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.623585</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.619160</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens)</td>
      <td>4.614540</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Super Mario Bros.</td>
      <td>4.610185</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>Witness</td>
      <td>4.601180</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.596695</td>
    </tr>
    <tr>
      <th>246</th>
      <td>Thelma &amp; Louise</td>
      <td>4.595952</td>
    </tr>
    <tr>
      <th>904</th>
      <td>Seven Samurai (The Magnificent Seven) (Shichin...</td>
      <td>4.595246</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>Queen Margot (La Reine Margot)</td>
      <td>4.591021</td>
    </tr>
    <tr>
      <th>928</th>
      <td>Ferris Bueller's Day Off</td>
      <td>4.576107</td>
    </tr>
    <tr>
      <th>1223</th>
      <td>Ghostbusters</td>
      <td>4.573215</td>
    </tr>
    <tr>
      <th>930</th>
      <td>Pleasantville</td>
      <td>4.571613</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cpost = ContextPostFiltering(sample, ['gender', 'age', 'occupations'], (1, 5))
cpost.fit([0, 2, 4])
cpost_recommendations1 = cpost.show_top_k(1406)
recom_table = final.loc[[i for i in cpost_recommendations1.index], ['title']]
recom_table['ratings'] = [cpost_recommendations1[i] for 
                          i in cpost_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 24. Jose's Recommendations - Gender, Age & Occupation Filter 
(Male, 25-34 years old, Grad Student)
</b></p>'''))

display(recom_table)
```

    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<p style="font-size:12px;font-style:default;"><b>
Table 24. Jose's Recommendations - Gender, Age & Occupation Filter 
(Male, 25-34 years old, Grad Student)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>954</th>
      <td>Payback</td>
      <td>4.883349</td>
    </tr>
    <tr>
      <th>3470</th>
      <td>Poltergeist</td>
      <td>4.881376</td>
    </tr>
    <tr>
      <th>906</th>
      <td>Super Mario Bros.</td>
      <td>4.794592</td>
    </tr>
    <tr>
      <th>2186</th>
      <td>Star Trek: First Contact</td>
      <td>4.715575</td>
    </tr>
    <tr>
      <th>1046</th>
      <td>Carrie</td>
      <td>4.709644</td>
    </tr>
    <tr>
      <th>930</th>
      <td>Pleasantville</td>
      <td>4.708762</td>
    </tr>
    <tr>
      <th>2708</th>
      <td>Flubber</td>
      <td>4.682580</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Blast from the Past</td>
      <td>4.671838</td>
    </tr>
    <tr>
      <th>3730</th>
      <td>Being There</td>
      <td>4.632394</td>
    </tr>
    <tr>
      <th>3435</th>
      <td>Queen Margot (La Reine Margot)</td>
      <td>4.629279</td>
    </tr>
    <tr>
      <th>858</th>
      <td>101 Dalmatians</td>
      <td>4.626997</td>
    </tr>
    <tr>
      <th>922</th>
      <td>One Flew Over the Cuckoo's Nest</td>
      <td>4.623585</td>
    </tr>
    <tr>
      <th>1759</th>
      <td>Nosferatu (Nosferatu, eine Symphonie des Grauens)</td>
      <td>4.614540</td>
    </tr>
    <tr>
      <th>3224</th>
      <td>Witness</td>
      <td>4.601180</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>Forbidden Planet</td>
      <td>4.600113</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>Scream 3</td>
      <td>4.594842</td>
    </tr>
    <tr>
      <th>2920</th>
      <td>Snow White and the Seven Dwarfs</td>
      <td>4.560849</td>
    </tr>
    <tr>
      <th>3271</th>
      <td>Swingers</td>
      <td>4.548405</td>
    </tr>
    <tr>
      <th>1212</th>
      <td>Raising Arizona</td>
      <td>4.542249</td>
    </tr>
    <tr>
      <th>1927</th>
      <td>Butch Cassidy and the Sundance Kid</td>
      <td>4.539263</td>
    </tr>
  </tbody>
</table>
</div>



```python
# setup spark
config = pyspark.SparkConf().setAll([('spark.driver.memory', '12g')])
sc = SparkContext(conf=config)
spark = SparkSession.builder.config(conf=config).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
```

#### Contextual Filtering

<p style="text-align:justify">In using contextual modeling, context is integrated into the model. Due to it being computationally expensive it was performed in Spark. The results below were also from a three filter model (age, gender & occupation). Interestingly, the recommendations here actually have no overlaps with the results from any post-filtering model used.</p>


```python
# recommendations with context filters

cm = ContextualModeling(sample, ['gender', 'age', 'occupations'])
cm.fit()
cm_recommendations1 = cm.show_top_k(3388, [1, 2, 1])
recom_table = final.loc[[i for i in cm_recommendations1.index], ['title']]
recom_table['ratings'] = [cm_recommendations1['prediction'][i] for 
                          i in cm_recommendations1.index]

display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 25. Maria's Recommendations - Gender, Age & Occupation Filter 
(Female, 25-34 years old, Academe)
</b></p>'''))
display(recom_table)
```

                                                                                    


<p style="font-size:12px;font-style:default;"><b>
Table 25. Maria's Recommendations - Gender, Age & Occupation Filter 
(Female, 25-34 years old, Academe)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>197</th>
      <td>Larger Than Life</td>
      <td>1.792370</td>
    </tr>
    <tr>
      <th>87</th>
      <td>Simon Birch</td>
      <td>1.790603</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>Ben-Hur</td>
      <td>1.786599</td>
    </tr>
    <tr>
      <th>920</th>
      <td>D2: The Mighty Ducks</td>
      <td>1.786394</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>Man on the Moon</td>
      <td>1.781901</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>Breakfast Club, The</td>
      <td>1.781469</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>Alien Nation</td>
      <td>1.780635</td>
    </tr>
    <tr>
      <th>1704</th>
      <td>Some Like It Hot</td>
      <td>1.780341</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>In the Line of Fire</td>
      <td>1.779848</td>
    </tr>
    <tr>
      <th>331</th>
      <td>Hook</td>
      <td>1.779605</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Close Shave, A</td>
      <td>1.779031</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Broken Arrow</td>
      <td>1.777247</td>
    </tr>
    <tr>
      <th>1209</th>
      <td>Shawshank Redemption, The</td>
      <td>1.776284</td>
    </tr>
    <tr>
      <th>580</th>
      <td>Professional, The (a.k.a. Leon: The Professional)</td>
      <td>1.775985</td>
    </tr>
    <tr>
      <th>986</th>
      <td>Little Mermaid, The</td>
      <td>1.775710</td>
    </tr>
    <tr>
      <th>1324</th>
      <td>Star Wars: Episode IV - A New Hope</td>
      <td>1.775609</td>
    </tr>
    <tr>
      <th>1614</th>
      <td>Old Yeller</td>
      <td>1.775402</td>
    </tr>
    <tr>
      <th>1579</th>
      <td>Alive</td>
      <td>1.774727</td>
    </tr>
    <tr>
      <th>330</th>
      <td>13th Warrior, The</td>
      <td>1.774110</td>
    </tr>
    <tr>
      <th>927</th>
      <td>Beauty and the Beast</td>
      <td>1.773974</td>
    </tr>
  </tbody>
</table>
</div>



```python
# recommendations with context filters

cm = ContextualModeling(sample, ['gender', 'age', 'occupations'])
cm.fit()
cm_recommendations1 = cm.show_top_k(1406, [0, 2, 4])
recom_table = final.loc[[i for i in cm_recommendations1.index], ['title']]
recom_table['ratings'] = [cm_recommendations1['prediction'][i] for 
                          i in cm_recommendations1.index]
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 26. Jose's Recommendations - Gender, Age & Occupation Filter 
(Male, 25-34 years old, Grad Student)
</b></p>'''))
display(recom_table)
```

                                                                                    


<p style="font-size:12px;font-style:default;"><b>
Table 26. Jose's Recommendations - Gender, Age & Occupation Filter 
(Male, 25-34 years old, Grad Student)
</b></p>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>ratings</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>882</th>
      <td>Last Emperor, The</td>
      <td>1.364233</td>
    </tr>
    <tr>
      <th>1064</th>
      <td>Christmas Vacation</td>
      <td>1.362749</td>
    </tr>
    <tr>
      <th>2487</th>
      <td>Austin Powers: The Spy Who Shagged Me</td>
      <td>1.360718</td>
    </tr>
    <tr>
      <th>2520</th>
      <td>Amistad</td>
      <td>1.360351</td>
    </tr>
    <tr>
      <th>938</th>
      <td>Toy Story</td>
      <td>1.360330</td>
    </tr>
    <tr>
      <th>2</th>
      <td>My Fair Lady</td>
      <td>1.360021</td>
    </tr>
    <tr>
      <th>1935</th>
      <td>Good, The Bad and The Ugly, The</td>
      <td>1.359694</td>
    </tr>
    <tr>
      <th>622</th>
      <td>Beauty and the Beast</td>
      <td>1.358498</td>
    </tr>
    <tr>
      <th>1436</th>
      <td>Girl, Interrupted</td>
      <td>1.357950</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Princess Bride, The</td>
      <td>1.357944</td>
    </tr>
    <tr>
      <th>2228</th>
      <td>Excalibur</td>
      <td>1.357559</td>
    </tr>
    <tr>
      <th>26</th>
      <td>E.T. the Extra-Terrestrial</td>
      <td>1.357236</td>
    </tr>
    <tr>
      <th>333</th>
      <td>Hamlet</td>
      <td>1.357225</td>
    </tr>
    <tr>
      <th>649</th>
      <td>Star Wars: Episode VI - Return of the Jedi</td>
      <td>1.357109</td>
    </tr>
    <tr>
      <th>1422</th>
      <td>Ben-Hur</td>
      <td>1.357095</td>
    </tr>
    <tr>
      <th>1499</th>
      <td>Raising Arizona</td>
      <td>1.356576</td>
    </tr>
    <tr>
      <th>867</th>
      <td>Crow: City of Angels, The</td>
      <td>1.356533</td>
    </tr>
    <tr>
      <th>111</th>
      <td>American Beauty</td>
      <td>1.355743</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Lady and the Tramp</td>
      <td>1.355644</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>Crocodile Dundee II</td>
      <td>1.355053</td>
    </tr>
  </tbody>
</table>
</div>


### C. Content based + Context aware recommendation

<p style="text-align:justify">Now we explore on mixing the results of content and context that we got. Perhaps, Maria wanted to explore movies that are on her genre while also having a female context of the movie. We did this by first normalizing the rating that we got for content based and post filtering context aware then perform a weighted sum, sort the order in terms of the added ratings in descending manner for the recommendations.</p>

#### Maria's Recommendations


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 27. Maria's Recommended movies based on content metadata 
and gender context
</b></p>'''))

# recommendations with context filters
cpost = ContextPostFiltering(sample, ['gender'], (1, 5))
cpost.fit([1])
cpost_recommendations1 = cpost.show_top_k(3388, top_k=1000)
post = pd.DataFrame(cpost_recommendations1)
post.columns = ['post_ratings']

# content
df_movies, X_new = preprocess_movies(df_movies)
cont = content_recom(3388, df, df_movies, X_new, top_k=100)
cont = cont.set_index('movieid')

mix = pd.merge(cont, post, left_index=True, right_index=True)

mix['rating'] = (mix.rating-mix.rating.min()
                  /(mix.rating.max()-mix.rating.min()))
mix['post_ratings'] = ((mix.post_ratings-mix.post_ratings.min())
                       /(mix.post_ratings.max()-mix.post_ratings.min()))
mix['mix_ratings'] = mix.rating * 0.5 + mix.post_ratings * 0.5

display(mix.sort_values(by='mix_ratings', 
                        ascending=False)[:20].iloc[:, [0, 6]])
```


<p style="font-size:12px;font-style:default;"><b>
Table 27. Maria's Recommended movies based on content metadata 
and gender context
</b></p>


    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3470</th>
      <td>3388</td>
      <td>Dersu Uzala</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>3388</td>
      <td>Winnie the Pooh and the Blustery Day</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3388</td>
      <td>Toy Story</td>
    </tr>
    <tr>
      <th>3114</th>
      <td>3388</td>
      <td>Toy Story 2</td>
    </tr>
    <tr>
      <th>2139</th>
      <td>3388</td>
      <td>Secret of NIMH, The</td>
    </tr>
    <tr>
      <th>1022</th>
      <td>3388</td>
      <td>Cinderella</td>
    </tr>
    <tr>
      <th>3034</th>
      <td>3388</td>
      <td>Robin Hood</td>
    </tr>
    <tr>
      <th>13</th>
      <td>3388</td>
      <td>Balto</td>
    </tr>
    <tr>
      <th>2096</th>
      <td>3388</td>
      <td>Sleeping Beauty</td>
    </tr>
    <tr>
      <th>1033</th>
      <td>3388</td>
      <td>Fox and the Hound, The</td>
    </tr>
    <tr>
      <th>1907</th>
      <td>3388</td>
      <td>Mulan</td>
    </tr>
    <tr>
      <th>596</th>
      <td>3388</td>
      <td>Pinocchio</td>
    </tr>
    <tr>
      <th>595</th>
      <td>3388</td>
      <td>Beauty and the Beast</td>
    </tr>
    <tr>
      <th>2018</th>
      <td>3388</td>
      <td>Bambi</td>
    </tr>
    <tr>
      <th>3159</th>
      <td>3388</td>
      <td>Fantasia 2000</td>
    </tr>
    <tr>
      <th>2138</th>
      <td>3388</td>
      <td>Watership Down</td>
    </tr>
    <tr>
      <th>594</th>
      <td>3388</td>
      <td>Snow White and the Seven Dwarfs</td>
    </tr>
    <tr>
      <th>364</th>
      <td>3388</td>
      <td>Lion King, The</td>
    </tr>
    <tr>
      <th>2099</th>
      <td>3388</td>
      <td>Song of the South</td>
    </tr>
    <tr>
      <th>2085</th>
      <td>3388</td>
      <td>101 Dalmatians</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">The recommended movies for maria does not match with the movies that we got for content and context but we got a similar genre for the profile of Maria which is adventure, children, and animation. Some of the recommended movies are wiinie the pooh, Dersu Uzala, and toy story</p>


#### Jose's Recommendations


```python
display(HTML('''<p style="font-size:12px;font-style:default;"><b>
Table 28. Jose's Recommended movies based on content metadata 
and gender context
</b></p>'''))

# recommendations with context filters
cpost = ContextPostFiltering(sample, ['gender'], (1, 5))
cpost.fit([1])
cpost_recommendations1 = cpost.show_top_k(3388, top_k=1000)
post = pd.DataFrame(cpost_recommendations1)
post.columns = ['post_ratings']

# content
df_movies, X_new = preprocess_movies(df_movies)
cont = content_recom(1406, df, df_movies, X_new, top_k=100)
cont = cont.set_index('movieid')

mix = pd.merge(cont, post, left_index=True, right_index=True)

mix['rating'] = (mix.rating-mix.rating.min()
                  /(mix.rating.max()-mix.rating.min()))
mix['post_ratings'] = ((mix.post_ratings-mix.post_ratings.min())
                       /(mix.post_ratings.max()-mix.post_ratings.min()))
mix['mix_ratings'] = mix.rating * 0.5 + mix.post_ratings * 0.5

display(mix.sort_values(by='mix_ratings', 
                        ascending=False)[:20].iloc[:, [0, 6]])
```


<p style="font-size:12px;font-style:default;"><b>
Table 28. Jose's Recommended movies based on content metadata 
and gender context
</b></p>


    Computing the pearson similarity matrix...
    Done computing similarity matrix.
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1084</th>
      <td>1406</td>
      <td>Bonnie and Clyde</td>
    </tr>
    <tr>
      <th>908</th>
      <td>1406</td>
      <td>North by Northwest</td>
    </tr>
    <tr>
      <th>3147</th>
      <td>1406</td>
      <td>Green Mile, The</td>
    </tr>
    <tr>
      <th>3362</th>
      <td>1406</td>
      <td>Dog Day Afternoon</td>
    </tr>
    <tr>
      <th>3517</th>
      <td>1406</td>
      <td>Bells, The</td>
    </tr>
    <tr>
      <th>1842</th>
      <td>1406</td>
      <td>Illtown</td>
    </tr>
    <tr>
      <th>2438</th>
      <td>1406</td>
      <td>Outside Ozona</td>
    </tr>
    <tr>
      <th>3019</th>
      <td>1406</td>
      <td>Drugstore Cowboy</td>
    </tr>
    <tr>
      <th>3936</th>
      <td>1406</td>
      <td>Phantom of the Opera, The</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>1406</td>
      <td>Sling Blade</td>
    </tr>
    <tr>
      <th>149</th>
      <td>1406</td>
      <td>Amateur</td>
    </tr>
    <tr>
      <th>2268</th>
      <td>1406</td>
      <td>Few Good Men, A</td>
    </tr>
    <tr>
      <th>1945</th>
      <td>1406</td>
      <td>On the Waterfront</td>
    </tr>
    <tr>
      <th>290</th>
      <td>1406</td>
      <td>Once Were Warriors</td>
    </tr>
    <tr>
      <th>3783</th>
      <td>1406</td>
      <td>Croupier</td>
    </tr>
    <tr>
      <th>3735</th>
      <td>1406</td>
      <td>Serpico</td>
    </tr>
    <tr>
      <th>3510</th>
      <td>1406</td>
      <td>Frequency</td>
    </tr>
    <tr>
      <th>850</th>
      <td>1406</td>
      <td>Cyclo</td>
    </tr>
    <tr>
      <th>3328</th>
      <td>1406</td>
      <td>Ghost Dog: The Way of the Samurai</td>
    </tr>
    <tr>
      <th>2963</th>
      <td>1406</td>
      <td>Joe the King</td>
    </tr>
  </tbody>
</table>
</div>


<p style="text-align:justify">Similarly for Jose, We got a match with genre of drama and crime. And some of the recommended movies are song of freedom, twenty four seven, and mr smith goes to washington</p>

<h2><font color='darkorange'> VI. Conclusion </font></h2>

<p style="text-align:justify">Several insights from the initial analysis of the data such as the most common user categories helped the team pick profiles to test in this study.</p>
    
<p style="text-align:justify">Using content based recommender systems highlight the similarity of the recommended movies to the viewing and rating history of the user. As evident in Maria's case, her profile as a middle aged female academician doesn't necessarily match or reflect in her viewing history which is dominated by children's shows. In this case the content based recommender system still gave suggestions that will appeal to her children. Meanwhile for Jose, the results also reflected his viewing history which is mostly drama. The content based system's recommendations were more of the same, which is good for the user if they do not want to explore other genre.</p>

<p style="text-align:justify">Meawnhile for context aware systems, the recommendations made are more influenced by the filters set instead of the viewing or rating history which was the opposite of the pre-filtering method. For Maria's case, the recommendations were no where near children appropriate content, and were more influenced by the age and gender context. </p>
    
<p style="text-align:justify">As for the disadvantages, contextual pre-filtering limits the movie choices even before trying to create a rating matrix so sometimes the wrong set of filters could make the system fail. That is not the case for contextual post filtering where we saw work for all types and combinations of filters. The same could be said for contextual modeling.</p>

<p style="text-align:justify">Combining the two systems by assigning equal weights to the results of the content based and context post-filtering generated the best results as it takes into account the theme and the filters to give a more varied roster of recommendations for the users.</p>

<h2><font color='darkorange'> VII. Recommendations </font></h2>

<p style="text-align:justify">Our recommendations to further improve our study in terms of data source is to try a larger dataset. In kaggle they also have Movielens 25M dataset on which the future researchers can explore. If available, they can use the images of the movies as a content metadata as well. In terms of models used,  We also recommend to try other models like Random Forest or Gradientboost for the Contextual modeling. Future researcher can also use deep learning models for recommender system for a very large datasets. And lastly, in terms of evaluation, we recommend to evaluate the models trained by performing a train and validation split. Tune the hyperparameters until you got the lowest error metric like Mean Absolute Error (MAE) or Mean Squared Error (MSE). Lastly, in order to fully validate the results of our model, we recommend to monitor the behavior of the user on which we gave our recommendation. If they liked our recommendation then it means our recommender system performed well otherwise there are some improvements that can be made in order to satisfy our user.</p>

<h2><font color='darkorange'> References </font></h2>

[1] White, S. (2021) "Streaming: How Digital Media is Transforming the Film Industry". https://medium.com/digital-society/streaming-how-digital-media-is-transforming-the-film-industry-1643aadb36ef<br><br>
[2] Abhijit, R. (2020). "Introduction To Recommender Systems- 1: Content-Based Filtering And Collaborative Filtering". https://towardsdatascience.com/introduction-to-recommender-systems-1-971bd274f421 <br><br>
[3] Espinosa, A. (2017). "The basics of Context-Aware Recommendations". https://medium.com/@andresespinosapc/the-basics-of-context-aware-recommendations-5dd7a939049b
