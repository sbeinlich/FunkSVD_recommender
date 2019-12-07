import surprise

# 1000 users 1700 movies 100k ratings
# Load the movielens dataset with 100k ratings of the form
# user id | movie id | rating | timestamp 
data = surprise.Dataset.load_builtin('ml-100k')

# Use the SVD algorithm.
algorithm = surprise.SVD()

# sample random trainset and testset
# test set is made of 25% of the ratings.
trainset, testset = surprise.model_selection.train_test_split(data, test_size=.25)

# Train the algorithm on the trainset, and predict ratings for the testset
algorithm.fit(trainset)
predictions = algorithm.test(testset)

# Then compute RMSE
surprise.accuracy.rmse(predictions)

# Then compute MAE
surprise.accuracy.mae(predictions)

# Run 5-fold cross-validation and print result
# Calculate Root Mean Square Error, Mean absolute Error
surprise.model_selection.cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#  RMSE values close to 1 show  how concentrated the data is around the line of best fit
#  It is basically average error where big errors are heavily penalized (because they are squared).
#  Having an RMSE of 0 means that all our predictions are perfect. 
#  Having an RMSE of 0.5 means that in average, we are approximately 0.5 off with each prediction.
#  MAE is similar, except it doesn't heavily penalize big errors
