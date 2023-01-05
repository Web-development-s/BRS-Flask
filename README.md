# Book Recommender System
## Project Description
* An online book recommendation system based on Spark, Python Flask, and [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)
* The book recommendation system refers to https://github.com/jadianes/spark-movie-lens.
* Modify the data processing part to support [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).
* It is suitable for beginners to learn how to build a recommendation system. Other data are attached at the bottom of this article for reference and study.
* **If you think it is good, please give the project a star to support～～**

## Model-based [collaborative filtering](https://xuefenghuang.github.io/collaborate-filter/)book recommendation
This article implements a simple application for recommending books to users.
* 1. Recommended algorithm:

In our online book recommendation system, we borrow the training and prediction functions of Spark's ALS algorithm, and each time we receive new data, we update it into the training data set, and then update the model trained by ALS.

Suppose we have a set of users who have expressed a preference for a set of books. The higher the user's liking for a book, the higher the rating it will give it, on a scale from 1 to 5. Let's visualize it as a matrix, with rows representing users and columns representing books. A user's rating for a book. All ratings range from 1 to 5, with 5 being the most liked. The first user (row 1) rated the first book (column 1) as 4. Empty cells indicate that the user has not rated the book.
![Image of Example1](https://github.com/XuefengHuang/spark-book-recommender-system/blob/master/images/example1.png)

Matrix factorization (e.g. SVD, SVD++) transforms both items and users into the same latent space, which represents the latent interactions between users and items. The rationale behind matrix factorization is that latent features represent how users rate items. Given the underlying description of a user and an item, we can predict how much a user will rate an item that has not yet been rated.

![Image of Example1](https://github.com/XuefengHuang/spark-book-recommender-system/blob/master/images/example2.png)

* 2. Data description:
Scoring data file:

`"User-ID";"ISBN";"Book-Rating"`

```
"276725";"034545104X";"0"
"276726";"0155061224";"5"
"276727";"0446520802";"0"
"276729";"052165615X";"3"
"276729";"0521795028";"6"
"276733";"2080674722";"0"
"276736";"3257224281";"8"
```

Book data file:

`"ISBN";"Book-Title";"Book-Author";"Year-Of-Publication";"Publisher";"Image-URL-S";"Image-URL-M";"Image-URL-L"`

```
"0195153448";"Classical Mythology";"Mark P. O. Morford";"2002";"Oxford University Press";"http://images.amazon.com/images/P/0195153448.01.THUMBZZZ.jpg";"http://images.amazon.com/images/P/0195153448.01.MZZZZZZZ.jpg";"http://images.amazon.com/images/P/0195153448.01.LZZZZZZZ.jpg"
"0002005018";"Clara Callan";"Richard Bruce Wright";"2001";"HarperFlamingo Canada";"http://images.amazon.com/images/P/0002005018.01.THUMBZZZ.jpg";"http://images.amazon.com/images/P/0002005018.01.MZZZZZZZ.jpg";"http://images.amazon.com/images/P/0002005018.01.LZZZZZZZ.jpg"
"0060973129";"Decision in Normandy";"Carlo D'Este";"1991";"HarperPerennial";"http://images.amazon.com/images/P/0060973129.01.THUMBZZZ.jpg";"http://images.amazon.com/images/P/0060973129.01.MZZZZZZZ.jpg";"http://images.amazon.com/images/P/0060973129.01.LZZZZZZZ.jpg"
"0374157065";"Flu: The Story of the Great Influenza Pandemic of 1918 and the Search for the Virus That Caused It";"Gina Bari Kolata";"1999";"Farrar Straus Giroux";"http://images.amazon.com/images/P/0374157065.01.THUMBZZZ.jpg";"http://images.amazon.com/images/P/0374157065.01.MZZZZZZZ.jpg";"http://images.amazon.com/images/P/0374157065.01.LZZZZZZZ.jpg"
"0393045218";"The Mummies of Urumchi";"E. J. W. Barber";"1999";"W. W. Norton &amp; Company";"http://images.amazon.com/images/P/0393045218.01.THUMBZZZ.jpg";"http://images.amazon.com/images/P/0393045218.01.MZZZZZZZ.jpg";"http://images.amazon.com/images/P/0393045218.01.LZZZZZZZ.jpg"
```

* 3. Data processing details:

Since the ISBN in the data is in string format, and the default product id of Spark’s ALS is in int format, the ISBN number is calculated and hashed and the first 8 digits are taken to prevent the integer from crossing the boundary. The detailed code is as follows:

```
dataset_path = os.path.join('datasets', 'BX-CSV-Dump')
sc = SparkContext("local[*]", "Test")
ratings_file_path = os.path.join(dataset_path, 'BX-Book-Ratings.csv')
ratings_raw_RDD = sc.textFile(ratings_file_path)
ratings_raw_data_header = ratings_raw_RDD.take(1)[0]
ratings_RDD = ratings_raw_RDD.filter(lambda line: line!=ratings_raw_data_header)\
            .map(lambda line: line.split(";")).map(lambda tokens: (int(tokens[0][1:-1]), abs(hash(tokens[1][1:-1])) % (10 ** 8),float(tokens[2][1:-1]))).cache()

books_file_path = os.path.join(dataset_path, 'BX-Books.csv')
books_raw_RDD = sc.textFile(books_file_path)
books_raw_data_header = books_raw_RDD.take(1)[0]
books_RDD = books_raw_RDD.filter(lambda line: line!=books_raw_data_header)\
    .map(lambda line: line.split(";"))\
    .map(lambda tokens: (abs(hash(tokens[0][1:-1])) % (10 ** 8), tokens[1][1:-1], tokens[2][1:-1], tokens[3][1:-1], tokens[4][1:-1], tokens[5][1:-1])).cache()
books_titles_RDD = books_RDD.map(lambda x: (int(x[0]), x[1], x[2], x[3], x[4], x[5])).cache()
```

* 4. Choose model parameters:
```
from pyspark.mllib.recommendation import ALS
import math

seed = 5L
iterations = 10
regularization_parameter = 0.1
ranks = [4, 8, 12]
errors = [0, 0, 0]
err = 0
tolerance = 0.02

min_error = float('inf')
best_rank = -1
best_iteration = -1
for rank in ranks:
    model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                      lambda_=regularization_parameter)
    predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
    error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1])**2).mean())
    errors[err] = error
    err += 1
    print 'For rank %s the RMSE is %s' % (rank, error)
    if error < min_error:
        min_error = error
        best_rank = rank

print 'The best model was trained with rank %s' % best_rank
```

* 5. Model Save
```
from pyspark.mllib.recommendation import MatrixFactorizationModel

model_path = os.path.join('..', 'models', 'book_als')

# Save and load model
model.save(sc, model_path)
same_model = MatrixFactorizationModel.load(sc, model_path)
```

* 6.Running instructions:
```
virtualenv book
pip install -r requirements.txt
python server.py
```

* 7. API:
```
GET: /<int:user_id>/ratings/top/<int:count> Obtain top N information of user book recommendations
GET: /<int:user_id>/ratings/<string:book_id> Get the user's evaluation information for a book
POST: /<int:user_id>/ratings Add book review information
```

* 8. 接口调用示例：


```
GET: /276729/ratings/top/3 Get the recommended top3 information of the book whose user ID is 276729
returned messages:

[
  {
    "Count": 30,
    "Rating": 8.781754720405482,
    "Author": "MARJANE SATRAPI",
    "URL": "http://images.amazon.com/images/P/0375422307.01.THUMBZZZ.jpg",
    "Publisher": "Pantheon",
    "Title": "Persepolis : The Story of a Childhood (Alex Awards (Awards))",
    "Year": "2003"
  },
  {
    "Count": 31,
    "Rating": 7.093566643463471,
    "Author": "Stephen King",
    "URL": "http://images.amazon.com/images/P/067081458X.01.THUMBZZZ.jpg",
    "Publisher": "Viking Books",
    "Title": "The Eyes of the Dragon",
    "Year": "1987"
  },
  {
    "Count": 25,
    "Rating": 7.069147186199548,
    "Author": "Jean Sasson",
    "URL": "http://images.amazon.com/images/P/0967673747.01.THUMBZZZ.jpg",
    "Publisher": "Windsor-Brooke Books",
    "Title": "Princess: A True Story of Life Behind the Veil in Saudi Arabia",
    "Year": "2001"
  }
]
```

```
GET: /276729/ratings/0446520802 Obtain user 276729's evaluation information on the book (ISBN: 0446520802)
returned messages:

[
  {
    "Count": 116,
    "Rating": 1.4087434932956826,
    "Author": "Nicholas Sparks",
    "URL": "http://images.amazon.com/images/P/0446520802.01.THUMBZZZ.jpg",
    "Publisher": "Warner Books",
    "Title": "The Notebook",
    "Year": "1996"
  }
]
```
## Other dataset recommendations (refer to https://gist.github.com/entaroadun/1653794)

The following data can be provided for beginners to learn how to train recommendation algorithm models

**Movie Data**:

* *MovieLens* - Movie Recommendation Data Sets http://www.grouplens.org/node/73
* *Yahoo!* - Movie, Music, and Images Ratings Data Sets http://webscope.sandbox.yahoo.com/catalog.php?datatype=r
* *Cornell University* - Movie-review data for use in sentiment-analysis experiments http://www.cs.cornell.edu/people/pabo/movie-review-data/

**Music data**:

* *Last.fm* - Music Recommendation Data Sets http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html
* *Yahoo!* - Movie, Music, and Images Ratings Data Sets http://webscope.sandbox.yahoo.com/catalog.php?datatype=r
* *Audioscrobbler* - Music Recommendation Data Sets http://www-etud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html
* *Amazon* - Audio CD recommendations http://131.193.40.52/data/


**Book data**:

* *Institut für Informatik, Universität Freiburg* - Book Ratings Data Sets http://www.informatik.uni-freiburg.de/~cziegler/BX/


**Gourmet data**:

* *Chicago Entree* - Food Ratings Data Sets http://archive.ics.uci.edu/ml/datasets/Entree+Chicago+Recommendation+Data


**Commodity data**:

* *Amazon* - Product Recommendation Data Sets http://131.193.40.52/data/


**Health data**:

* *Nursing Home* - Provider Ratings Data Set http://data.medicare.gov/dataset/Nursing-Home-Compare-Provider-Ratings/mufm-vy8d
* *Hospital Ratings* - Survey of Patients Hospital Experiences http://data.medicare.gov/dataset/Survey-of-Patients-Hospital-Experiences-HCAHPS-/rj76-22dk


**Dating data**:

* *www.libimseti.cz* - Dating website recommendation (collaborative filtering) http://www.occamslab.com/petricek/data/


**Academic article recommendation**:

* *National University of Singapore* - Scholarly Paper Recommendation http://www.comp.nus.edu.sg/~sugiyama/SchPaperRecData.html
