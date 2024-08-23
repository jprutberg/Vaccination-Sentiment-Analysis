# Vaccination Sentiment Analysis Using LLM, NLP, and Time Series Techniques

## Introduction
This project focuses on analyzing public sentiment toward COVID-19 vaccination using a combination of Natural Language Processing (NLP) techniques and time series analysis. Leveraging Large Language Models (LLMs) for advanced text processing, the analysis is conducted on a dataset of social media posts to understand sentiment trends over time. The project utilizes several Python libraries, including `pandas` for data manipulation, `numpy` for numerical operations, `TextBlob` for sentiment analysis, `matplotlib` and `seaborn` for data visualization, and `statsmodels` and `pmdarima` for time series analysis and forecasting.

## Purpose
The primary purpose of this project is to develop a model that can accurately assess and forecast public sentiment on COVID-19 vaccination over time. By analyzing sentiment trends, the project aims to provide insights that can inform public health communication strategies and improve understanding of public attitudes toward vaccination.

## Significance
Public sentiment on vaccination is a critical factor in the success of immunization campaigns, particularly during global health crises like the COVID-19 pandemic. By applying advanced NLP techniques, LLMs, and time series forecasting, this project contributes to the understanding of how public opinion evolves over time and the factors that influence these changes. The findings can help public health officials and policymakers design more effective communication strategies that resonate with public sentiment and address concerns in real-time.

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
5. Model Evaluation: The performance of the ARIMA model was evaluated against the Holt-Winters Exponential Smoothing model using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

## Findings and Conclusions
### Sentiment Trends:
The analysis revealed key trends in public sentiment towards vaccination, with notable spikes and drops corresponding to significant events or announcements.

### Model Performance:
The ARIMA model slightly outperformed the Holt-Winters model in forecasting sentiment, suggesting that ARIMA may be better suited for this type of time series data. However, both models provided valuable insights into potential future sentiment trends.

### Public Health Implications:
Understanding these sentiment trends can help public health officials anticipate public reactions and tailor their communication strategies to maintain or improve public trust in vaccination programs.

## References
This project utilized Python for data analysis, with key libraries including `pandas`, `numpy`, `TextBlob`, `matplotlib`, `seaborn`, `statsmodels`, and `pmdarima`. The dataset used for this analysis is available on Kaggle at https://www.kaggle.com/datasets/gpreda/all-covid19-vaccines-tweets. Due to its large size, the CSV file is not included in this repository, but full code and further documentation are available.
