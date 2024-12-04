from GoogleNews import GoogleNews
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from datetime import datetime, date
import numpy as np

nltk.download('vader_lexicon')

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize text into words
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(words)

def process_text(search_word: str):
    # Download stopwords and punkt tokenizer if not already downloaded
    nltk.download('stopwords')
    nltk.download('punkt')

    gn = GoogleNews(period="7d")

    print(gn.getVersion())

    gn.enableException(True)

    gn.get_news(search_word)

    parsed_news = gn.get_texts()

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    columns = ['Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    values = []

    print(news)

    # print(f"news - {type(gn.results())} - {gn.results()}")
    for result in parsed_news:
        print(f"news - {type(result)} - {result}")
        # Clean and preprocess the text
        cleaned_text = preprocess_text(result)

        # mean = round(dataframe['compound'].mean(), 2)
        # values.append(mean)

        print(f"cleaned_text - {cleaned_text} - score: {scores}")


def process_text_attempt_2():
    # Parameters
    n = 3
    tickers = ['AAPL', 'TSLA', 'AMZN']

    # Get Data
    finwiz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}

    for ticker in tickers:
        url = finwiz_url + ticker
        req = Request(url=url,headers={'User-Agent': 'Mozilla/5.0'})

        print(url)

        resp = urlopen(req)
        html = BeautifulSoup(resp, features="lxml")
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table

    try:
        for ticker in tickers:
            df = news_tables[ticker]
            df_tr = df.findAll('tr')

            print ('\n')
            print ('Recent News Headlines for {}: '.format(ticker))

            for i, table_row in enumerate(df_tr):
                a_text = table_row.a.text
                td_text = table_row.td.text
                td_text = td_text.strip()
                print(a_text,'(',td_text,')')
                if i == n-1:
                    break
    except KeyError:
        pass

    # Iterate through the news
    parsed_news = []
    for file_name, news_table in news_tables.items():
        for x in news_table.findAll('tr'):
            text = x.a.get_text()
            date_scrape = x.td.text.split()

            if len(date_scrape) == 1:
                time = date_scrape[0]

            else:
                date = date_scrape[0]
                time = date_scrape[1]

            ticker = file_name.split('_')[0]

            parsed_news.append([ticker, date, time, text])

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()

    columns = ['Ticker', 'Date', 'Time', 'Headline']
    news = pd.DataFrame(parsed_news, columns=columns)
    scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

    df_scores = pd.DataFrame(scores)
    news = news.join(df_scores, rsuffix='_right')

    # View Data
    news['Date'] = np.where(news['Date'] =="Today", datetime.today().strftime("%b-%d-%y"), news['Date'])
    news["cleaned_nlp"] = news["Headline"].apply(lambda x : preprocess_text(x))

    print(datetime.today().strftime("%Y-%m-%d"))
    print(news)

    news['Date'] = pd.to_datetime(news.Date).dt.date

    unique_ticker = news['Ticker'].unique().tolist()
    news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

    values = []
    for ticker in tickers:
        dataframe = news_dict[ticker]
        dataframe = dataframe.set_index('Ticker')
        dataframe = dataframe.drop(columns = ['Headline'])
        dataframe = dataframe.drop(columns = ['cleaned_nlp'])
        print ('\n')
        print (dataframe.head())

        mean = round(dataframe['compound'].mean(), 2)
        values.append(mean)

    df = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean Sentiment'])
    df = df.set_index('Ticker')
    df = df.sort_values('Mean Sentiment', ascending=False)
    print ('\n')
    print (df)


# process_text("ftse")

process_text_attempt_2()
