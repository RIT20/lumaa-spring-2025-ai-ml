# AI/Machine Learning Intern Challenge: Simple Content-Based Recommendation

**Deadline**: Sunday, Feb 23th 11:59 pm PST

---
Dataset: 
   I have used kaggle data, it is comprise of 16 different genre of data(in .csvs), I combined them and randomly selected 800 rows for our use case in sample_movies.csv
   link to dataset: https://www.kaggle.com/datasets/rajugc/imdb-movies-dataset-based-on-genre

   Valuable columns 
   'movie_id' -- ID, 
   'movie_name' -- Movie name, 
   'year' -- Year they released, 
   'genre' -- Genre they belong like action, comedy, romance etc. 
   'description' -- Movie description(helpful in the context of matching), 
   'director' -- Director name(),
   'star' -- Star name

Setup:
   I have added requirement.txt for running, used python 3 with latest version python 3.13.2. I believe normal setup with all the necessary library will be enough to run this project. 

Running:
   For running the code -- we have to run test.ipynb
   steps
   Read sample_movies.csv dataset, I have manipulated some column like adding the genre , star names, movie_name, director names and year, incase if someone wants to search for specific star, director or movie name. It can be run wthout this step as well.

   Then in the next cell Input query, where we can put our use case, and number of results we want and we can see final result in final_df

Results:
   Final_df is the Final_Dataset.
      Input_query -- User query we haev to put
      Tfidf_query -- Transformed query by vectorizer
      original_movie_description -- Original descrition of the movie
      featured_movie_description -- Some feature manipulation but used for vectorization in Tf-idf.
      cos_sim_score -- similarity score based on cosinie similarity score.

   For Example of one such case is given below
   Input : 'I like to see action movies' 
   Output : 'After a crime boss has Tony Quinn blinded by acid, Tony is given the ability to see in the dark like a bat....' 
            'In a utopian society created at the end of the third world war, a female warrior....'
            'Picking up after the events of the first film, Lock and Key press forward as their paths come dangerously...'
            'A young girl who returns to her hometown to see her dying father finds herself being drawn into a web...'

Salary expectation per month (Mandatory):
   $20 - $30 per hour/ $3200 - $4800 per month

Video Link :
   https://drive.google.com/file/d/1ud_nyaGfisFds7tm9PqcSSp7KQdjLIHT/view?usp=drive_link
