
**Stock Market Prediction Analysis**

**Abstract** - Prediction of stock market trends is considered as an important task and is of great attention as predicting stock prices successfully may lead to attractive profits by making proper decisions. Stock market prediction is a major challenge owing to non-stationary, blaring, and chaotic data, and thus, the prediction becomes challenging among the investors to invest the money for making profits. A number of techniques have already come up suggesting better prediction methods. In this research paper, we have used LSTM(Long short-term memory) model for predicting stock prices.

**Introduction**

Forecasting stock market prices has always been a challenging task for many business analysts and researchers. The stock market is known for being volatile, dynamic, and nonlinear. Accurate stock price prediction is extremely challenging because of multiple (macro and micro) factors, such as politics, global economic conditions, unexpected events, a company’s financial performance, and so on. There are several factors that not only influence the market directly but also affect other factors that in turn affect markets. All of these make the overall prediction of the stock market a complex task, not only due to the number of variables involved but also how these variables affect each other.

There are no guarantees of profits when you buy stock, which makes stock **one of the riskiest investments**. If a company doesn't do well or falls out of favour with investors, its stock can fall in price, and investors could lose money.

Risks involved:

Market risk

Interest rate risk

Company risk

Regulatory risk

Liquidity risk

Taxability risk

Inflation risk

There are some negative factors also that affect a particular business like redundant technology, unforeseen competition, or incompetent management. Some global and country-level economic factors like recession, war or trade war between countries may also negatively impact the stock market as a whole. At these times, the stocks would lose their value and the investors would lose their wealth. This is an inherent risk that the investors have to assume when they choose to invest in the stock market. They earn returns higher than those on low-risk instruments like bank fixed deposits but they also risk losing their entire investment. But, all of this also means that there’s a lot of data to find patterns in. So, financial analysts, researchers, and data scientists keep exploring analytics techniques to detect stock market trends. This gave rise to the concept of algorithmic training which uses automated, pre-programmed trading strategies to execute orders.


**Dataset used** - Google’s stock price data for the year 2017

Train dataset


Test dataset

**Literature survey**

|S.No|Title and publication|Core idea of the paper|Approaches used|Experimental Data|
| :- | :- | :- | :- | :- |
|**1.**|A Literature Survey on Stocks Predictions using Hybrid Machine Learning and Deep Learning Models|This paper is a survey on the application of neural networks in forecasting stock market prices. With their ability to discover patterns in nonlinear and chaotic systems, neural networks offer the ability to predict market directions more accurately than current techniques. Common market analysis techniques such as technical analysis, fundamental analysis, and regression are discussed and compared with neural network performance.|The multi-layer Perceptron (MLP), the Convolutional Neural Networks (CNN), and the Long ShortTerm Memory (LSTM) recurrent neural networks technique.Deep neural networks (DNNs) are powerful types of artificial neural networks (ANNs) that use several hidden layers|stock market dataset features high-level stock market data taken from the Nasdaq, Dow Jones, and S&P 500 market indexes beginning in 1977 and ending in 2017.|
|**2.**|Stock Market Prediction Using Machine Learning(ML)Algorithms|Stock Market Prediction; Machine Learning(ML); Algorithms; Linear Regression; Exponential Smoothing; Time Series Forecasting|Machine Learning Algorithm specially focus on Linear Regression (LR), Three month Moving Average(3MMA), Exponential Smoothing (ES) and Time Series Forecasting using MS Excel as best statistical tool for graph and tabular representation of prediction results.|data is obtained from Yahoo Finance for Amazon (AMZN) stock, AAPL stock and GOOGLE stock after implementation LR we successfully predicted stock market trend for next month and also measured accuracy according to measurements.|
|<p>**3.**</p><p></p>|Stock Market Analysis: A Review and Taxonomy of Prediction Techniques|Application of machine learning techniques and other algorithms for stock price analysis and forecasting is an area that shows great promise. In this paper, they provide a concise review of stock markets and taxonomy of stock market prediction methods. They then focus on some of the research achievements in stock analysis and prediction.|stock exchanges; stock markets; analysis; prediction; statistics; machine learning; pattern recognition; sentiment analysis|Korea Composite Stock Price Index,the German stock index or Deutscher Aktienindex (DAX) and the Financial Times Stock Exchange (FTSE|
|**4.**|Effectiveness of Artificial Intelligence in Stock Market Prediction Based on Machine Learning|This paper tries to address the problem of stock market prediction leveraging artificial intelligence (AI) strategies. The stock market prediction can be modeled based on two principal analyses called technical and fundamental. In the technical analysis approach, the regression machine learning (ML) algorithms are employed to predict the stock price trend at the end of a business day based on the historical price data. In contrast, in the fundamental analysis, the classification ML algorithms are applied to classify the public sentiment based on news and social media. In the technical analysis, the historical price data is exploited from Yahoo Finance, and in fundamental analysis, public tweets on Twitter associated with the stock market are investigated to assess the impact of sentiments on the stock market’s forecast. The results show a median performance, implying that with the current technology of AI, it is too soon to claim AI can beat the stock markets.|Machine learning, time series prediction, technical analysis, sentiment embedding, financial market.|the dataset includes some financial indicators such as RSI and MACD as features and the stock’s closing price as the target value in the technical analysis approach. It is evident that the data associated with the technical analysis is continuous numbers, which is shown in time-series format data|
|**5.**|Stock Market Prediction and Portfolio Management using ML techniques|<p>Uses online learning algorithm that utilizes a kind of recurrent neural network (RNN) called Long Short Term Memory (LSTM), where the weights are adjusted for individual data points using stochastic gradient descent. This will provide more accurate results when compared to existing stock price prediction algorithms.</p><p></p>|The network is trained and evaluated for accuracy with various sizes of data, and the results are tabulated. A comparison with respect to accuracy is then performed against an Artificial Neural Network.|datasets: Stock Prices obtained using Yahoo! Finance API. This dataset consists of the Open, Close, High and Low values for each day. • They obtained also a collection of tweets using Twitter’s Search API.|
|**6.**|Study on the prediction of stock price based on the associated network model of LSTM|It has proposed an associated deep recurrent neural network model with multiple inputs and multiple outputs based on long short-term memory network. The associated network model can predict the opening price, the lowest price and the highest price of a stock simultaneously. The associated network model was compared with LSTM network model and deep recurrent neural network model. The experiments show that the accuracy of the associated model is superior to the other two models in predicting multiple values at the same time, and its prediction accuracy is over 95%.|Deep learning · Machine learning · Long short-term memory (LSTM) · Deep recurrent neural network · Associated network|Three data sets were used in the experiments, one is Shanghai composite index 000001 and the others are two stocks of PetroChina (stock code 601857) on Shanghai stock exchange and ZTE (stock code 000063) on Shenzhen stock exchange. Shanghai composite index has 6112 historical data; PetroChina has 2688 historical data and ZTE has 4930 historical data.|

**Methodology**

Long Short Term Memory Network is an advanced RNN, a sequential network, that allows information to persist. It is capable of handling the vanishing gradient problem faced by RNN. A recurrent neural network is also known as RNN is used for persistent memory.

Let’s say while watching a video you remember the previous scene or while reading a book you know what happened in the earlier chapter. Similarly RNNs work, they remember the previous information and use it for processing the current input. The shortcoming of RNN is, they can not remember Long term dependencies due to vanishing gradient. LSTMs are explicitly designed to avoid long-term dependency problems.


Two methods that are widely used in general are namely Fundamental Analysis and Technical Analysis

Fundamental analysis - it has the ability to predict changes using parameters such as book value, earnings, p/e ratio, ROI. To determine accurate product value, reliable and accurate information on the ﬁnancial report of the company, it is necessary to have competitive strength and economic conditions in which they are interested. This also helps in investment decision as in if intrinsic value is higher than the market value it holds, invest otherwise and avoid it as a bad investment. Technical Analysis: “The idea behind technical analysis is that investors’ constantly changing attributes in response to different forces/factors make stock prices trends/movements”. Different technical factors of quantitative parameters can be used for analysis, such as trend indicators, lowest and highest daily values, indices, daily ups and downs, stock volume, etc. It is possible to extract rules from the data and the investors make future decisions based on these rules

We are using machine learning and deep learning techniques and will use the LSTM network to train our model with Google stocks data. LSTMs are recurrent neural networks used for learning long-term dependencies. Mainly used for processing and predicting time series data.

- Import libraries
- Load training dataset
- Use open stock price to train the model
- Normalize the dataset
- Create x\_train and y\_train data structures
- Reshape the dataset
- Building the model by importing the crucial libraries and adding different layers to LSTM
- Fitting the model
- Extracting the Actual Stock Prices
- Preparing the Input for the Model
- Predicting the Values\*\*\*\*

Plotting the Actual and Predicted Prices After predicting the results, we will be demonstrating the results using Tableau and storing it in an SQL database

