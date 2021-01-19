# 4A Trading
This project, 4A is a trading dashboard that enables retail investors to make an educated decision on whether to buy or sell a stock of their choice. The algorithm predicts future price movements of a stock using Natural Language Processing (NLP) sentiment analysis, Machine learning prediction.

## Libraries used
- os
- re
- ta
- json
- pandas
- numpy
- panel
- dotenv
- datetime
- warnings
- pathlib
- datetime
- yfinance
- finhub
- nltk
- newsapi
- tensorflow
- sklearn
- requests
- matplotlib
- hvplot
- tweepy
- alpaca_trade_api
- plotly.express
- boken.models
- nltk.sentiment.vader

## Tools used
- Alpaca
- News API
- Twitter API
- Stock news API
- Finhub

## Keys/tokens required prior to running
Assuming the above tools have already been installed, the following keys/ tokens are required to run the project notebooks.
- Alpaca Trade API (https://app.alpaca.markets/)
- Stock News API key (https://stocknewsapi.com/)
- News API Key (https://newsapi.org/)
- Twitter API key (https://developer.twitter.com/en/docs/twitter-api)
- Twitter secret key
- Twitter access token
- Twitter access secret token
- Twitter bearer token
- Finhub API key (https://finnhub.io/)
- Finhub Sandbox key 

## Analysis performed
- v3_stock_api_sentiment_analysis.ipynb: Stock news sentiment analysis and returns prediction
- twitter_ml_training.ipynb: Twitter sentiment analysis and returns prediction
- TechnicalPrep.ipynb: Technical analysis
- FundamentalPrep.ipynb: Fundamental analysis
- FinancialAnalysis_LR.ipynb: Price prediction using technical and fundamental analysis
- FinanceTab.ipynb
- main_analysis.ipynb: Panel Dashboard

## Data cleanup/ preparation
### News sentiment analysis
- One year's worth of stock data was pulled from Yahoo Finance
- The percentage returns was calculated.
- 1000+ items of news data was imported from Stock News API
- The news articles and sentiment was merged with the returns and compound sentiment scores
- For the linear regression, 10 months of training data and 3 monts of testing data was used
- For Deep Learning, the compound score was used as the feature and the returns was used as the target.
- 70% of the data was used for training and the rest for testing
- The training and the testing data was scaled and reshaped into vertical vector.
- After running the Neural Network (NN), the scaled data was inverse transformed to get the original.

### Twitter sentiment analysis
- The Twitter and Alpaca API are set up/authenticated
- The CSV file is imported, all non-alpha numeric characters are removed
- The tweets are fetched by the hour
- The normalized sentiment score is calculated.
- The raw tweets (by the hour) are saved to a dataframe

### Technical and fundamental analysis

## Metrics calculated
### News sentiment analysis
- Daily percentage returns
- Compound sentiment score
- Linear regression: Predicted return
- Deep Learning: Predicted return
- Normalized sentiment score (0 or 1)
- Accuracy score for Vader and RNN LSTM
- Confusion matrix for Vader and RNN LSTM
- Classification report for Vader and RNN LSTM
- Area Under the Curve (AUC) Vader and RNN LSTM

### Twitter sentiment analysis
- Compound sentiment score
- Correlation between closing price, returns and compound score
- Real vs predicted returns

### Technical and fundamental analysis

## Hyperparameters
### News sentiment analysis
> Price prediction:
 - Window: 1 day
 - Number of hidden units: 10
 - Dropout units: 0.1
 - Epochs: 10
 - Batch size: 5

> Sentiment analysis:
- Max words: 140
- Embedding size: 64
- Epochs: 10
- Batch size: 1000

### Twitter sentiment analysis
- Window: 5 days
- Number of hidden units: 5
- Dropout units: 0.1
- Epochs: 15
- Batch size: 2

### Technical and fundamental analysis

## Plots created
### News sentiment analysis
- Linear Regression: Real vs predicted returns
- Deep Learning: Real vs predicted returns
- AUC Vader
- AUC RNN LSTM

### Twitter sentiment analysis
- Twitter sentiment vs Daily percent change
- Real vs predicted returns

### Technical and fundamental analysis

## Interpretation
### News sentiment analysis
- The linear regression actual and predicted returns are inconsistent by a big margin. More training data could have been used.
- The deep learning predicted retruns falls flat, signifying that sufficient data has not been feeded for training.
- For the NLP sentiment prediction by Deep Learning, the accuracy score of Vader is perfect while the RNN LSTM score is only 46%. All the positive and negative sentiments from Vader have been correctly predicted, while all the negative sentiment scores from RNN LSTM have been missed by the model

### Twitter sentiment analysis
- From the Twitter sentiment vs Daily percent change graph, the sentiment score varies wildly with respect to the returns.
- Here too, the predicted retrun falls flat compared with the real returns

### Technical and fundamental analysis

## Panel Visualization
- The requisite notebooks are first loaded.
- The technical analysis, twitter sentiment and news sentiment dataframes are combined
- A function is defined to determine whether to buy, sell or hold the stock
- All the predictions are fetched and stored into dataframes.
- Panes for average combined price prediction and decision recommendation, as well as the individual predictions for technical, twitter and sentiment analysis are created.
- A panel dashboard is created with Welcome column that lets a user input a ticker of their choice and a single keyword; Individual predictions columns and Financial information

## Conclusion

## To run the main notebook
- Open the terminal and run `panel serve 4a_returns_predictor.ipynb` --show

## Difficulties/ Challenges faced
- The NEWS API free plan allows only 1 month of data to be pulled, while the stock news API limits to 100 calls under the free plan.
- Consequently, sufficient amount of data was not used to train the regression and ML models.
- For the Twitter API only 1 week's worth of data could be retreived.

## If there was more time
- A trial and error of the machine learning hyperparameters would have been performed.
- Also, additional neural network layers would have been tested to make the predictions as close as possible to the actual.
- An algorithmic trading strategy would have been developed and connected to a live trading platform

## Presentation Link
- https://docs.google.com/presentation/d/1lxMbRWSGchM7qw1O8nyFVtkrnmOeV-CiL7Z4Pkx-Gl0/edit?usp=sharing

## Contributors
- Albert Kong
- Kristofer Kish
- Satheesh Narasimman

## People who helped
- Allan Hall, Bootcamp Instructor
- Joel Gonzalez, Bootcamp Teaching Assistant
- Khaled Karman, Satheesh's Bootcamp Tutor

## References
- https://medium.com/automated-trading/obtain-40-technical-indicators-for-a-stock-using-python-247b32e85f30

- https://technical-analysis-library-in-python.readthedocs.io/en/latest/

- https://panel.holoviz.org/user_guide/Components.html

- https://stocknewsapi.com/