# Vaccination Sentiment Analysis Using LLM, NLP, Time Series, and RNN Techniques

## Introduction
This project focuses on analyzing public sentiment toward COVID-19 vaccination using a combination of Natural Language Processing (NLP) techniques, time series analysis, and Recurrent Neural Networks (RNNs).  Leveraging Large Language Models (LLMs) for advanced text processing, the analysis is conducted on a dataset of social media posts to understand sentiment trends over time. The project utilizes several Python libraries, including `pandas` for data manipulation, `numpy` for numerical operations, `TextBlob` for sentiment analysis, `matplotlib` and `seaborn` for data visualization, `statsmodels` and `pmdarima` for time series analysis and forecasting, and `PyTorch` for implementing RNNs.

## Purpose
The primary purpose of this project is to develop a model that can accurately assess and forecast public sentiment on COVID-19 vaccination over time. By analyzing sentiment trends and comparing different modeling approaches, including ARIMA, Holt-Winters, and RNNs, the project aims to provide insights that can inform public health communication strategies and improve understanding of public attitudes toward vaccination.

## Significance
Public sentiment on vaccination is a critical factor in the success of immunization campaigns, particularly during global health crises like the COVID-19 pandemic. By applying advanced NLP techniques, LLMs, time series forecasting, and RNNs, this project contributes to the understanding of how public opinion evolves over time and the factors that influence these changes. The findings can help public health officials and policymakers design more effective communication strategies that resonate with public sentiment and address concerns in real-time.

## Project Overview
### Data Collection:
The dataset consists of 228,207 tweets related to COVID-19 vaccination from December 2020 to November 2021. The data was preprocessed to remove noise and irrelevant content, ensuring that the analysis focuses on meaningful text. The data was sourced from a Kaggle dataset. Due to its large size, the CSV file is not uploaded to this GitHub repository.

### Text Processing and Sentiment Analysis:
1. Tokenization: Basic tokenization was performed using regular expressions to clean and prepare the text data for analysis.
2. Sentiment Analysis: Sentiment scores were assigned to each post using the `TextBlob` library, which provides a simple API for processing textual data and determining polarity.

### Time Series Analysis:
1. Sentiment Aggregation: Daily sentiment scores were aggregated to analyze trends over time.
2. Time Series Decomposition: Using the `seasonal_decompose` function from `statsmodels`, the sentiment time series was decomposed into trend, seasonal, and residual components.
3. Stationarity Testing: The Augmented Dickey-Fuller (ADF) test was applied using `adfuller` from `statsmodels` to check for stationarity in the sentiment time series.
4. ARIMA Modeling: The `auto_arima` function from `pmdarima` and `ARIMA` from `statsmodels` were employed to build and fit ARIMA models to the sentiment data. The dataset was split into a training set (the first 80% of observations) and a testing set (the last 20% of observations). This split was used for out-of-sample forecast validation, where the model was trained on the training data and its forecasting accuracy was validated on the unseen testing data. Forecasting was then performed to predict future sentiment trends.
5. Holt-Winters Modeling: The Holt-Winters Exponential Smoothing model was applied to compare its performance with ARIMA.
6. RNN Modeling: A Recurrent Neural Network (RNN) was implemented using `PyTorch` to forecast sentiment and compare its performance with traditional time series models.
7. Model Evaluation: The performance of all three models was compared using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

## Findings and Conclusions
### Sentiment Trends: 
The analysis uncovered significant patterns in public sentiment toward vaccination, highlighting how public opinion fluctuates in response to various events. These fluctuations underscore the sensitivity of public sentiment to external factors and the importance of timely, accurate communication from health authorities to maintain public confidence in vaccination efforts.

### Model Performance:
1. ARIMA: The ARIMA model provided a solid baseline for forecasting sentiment, with a Mean Squared Error (MSE) of 0.000338 and a Mean Absolute Percentage Error (MAPE) of 16.25%.
2. Holt-Winters: The Holt-Winters model was slightly less accurate than ARIMA, with an MSE of 0.000404 and a MAPE of 17.55%.
3. RNN: The RNN outperformed both ARIMA and Holt-Winters in terms of MSE and MAE, achieving an MSE of 0.000291 and a MAE of 0.012893. However, the MAPE of 16.93% was close to that of the ARIMA model, indicating strong performance in forecasting sentiment.

### Public Health Implications:
Understanding these sentiment trends can help public health officials anticipate public reactions and tailor their communication strategies to maintain or improve public trust in vaccination programs.

## References
This project utilized Python for data analysis, with key libraries including `pandas`, `numpy`, `TextBlob`, `matplotlib`, `seaborn`, `statsmodels`, `pmdarima`, and `PyTorch`. The dataset used for this analysis is available on Kaggle at https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets. Due to its large size, the CSV file is not included in this repository, but full code and further documentation are available.
