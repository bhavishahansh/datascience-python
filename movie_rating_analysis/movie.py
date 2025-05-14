import numpy as np
import pandas as pd
import plotly.express as px

movies = pd.read_csv("movies.dat", delimiter= '::')
movies.columns = ["ID","Title","Genre"]
#print(movies.head())

ratings = pd.read_csv("ratings.dat", delimiter= '::')
ratings.columns = ["User","ID","Ratings","Timestamp"]

data = pd.merge(movies,ratings,on=["ID", "ID"])
'''
value_counts() returns seriese and uses for quick counts,inspection
value_counts().reset_index() returns DataFrame and uses For plotting and DataFrame operations
'''
ratings_counts = data["Ratings"].value_counts().reset_index()
# renaming the columns of returned dataframe rating_counts
ratings_counts.columns  = ["Rating","Count"]
fig = px.pie(ratings_counts, values='Count', names='Rating', title='Ratings Distribution')
fig.show()
#print(ratings)

'''
As 10 is the highest rating a viewer can give, let’s take a look at the top 10 movies that got 10 ratings by viewers:

'''
data2 = data.query("Ratings == 10")
print(data2["Title"].value_counts().head(10))

# which movie had highest rating? do analysis
