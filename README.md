# Momentum trading VS Pairs trading: Which strategy is the most efficient ?
Based on Goulding and Al. paper "Decoding Decoding Systematic Relative Investing: A Pairs Approach", the idea of this project is to compare cross-sectional momentum (csmom) trading strategy performance and pairs trading strategy performance.

The inputs used to run this code is :
* Portfolio = "data_mef.csv" : 15 futures on the following equity indices of developed countries:
  - DAX, Germany (GX1 Index)
  - ASX SPI, Australia (XP1 Index)
  - SP TSX, Toronto (PT1 Index)
  - IBEX 35, Spain (IB1 Index)
  - TOPIX, Tokyo (TP1 Index)
  - Hang Seng, Hong Kong (HI1 Index)
  - FTSE MIB, Italy (ST1 Index)
  - STI, Singapore (SD1 Index)
  - OMX, Stockholm (QC1 Index)
  - SMI, Switzerland (Geneva, Zurich, Basel) (SM1 Index)
  - AEX, Amsterdam (EO1 Index)
  - FTSE 100, London (Z 1 Index)
  - SP 500, United States (SP1 Index)
  - CAC40, France (CF1 Index)
  - Euro Stoxx50, Eurozone (VG1 Index)

For each futures, the metric used is the closed price.
  
* Period before cleaning (cf clean_dataset function) : 29/12/1995 - 01/10/2021 
* Period after cleaning : 15/02/2005 - 01/10/2021

* Parameters of the strategies = "inputs_data.csv" : 
  - %train = split dataset to train (%train of dataset) and test (1-%train of dataset) datasets 
  - mom_duree = number of days/months/year of the momentum used to calculate signals
  - mom_type = type of momentum (days "d", months "m", year "y")
  - pond_long = long repartition of the portfolio for the momentum trading strategy
  - pond_short = short repartition of the portfolio for the momentum trading strategy
  - rebal_duree	= number of days/months/year of the rebalancing frequency
  - rebal_type = type of rebalancing frequency (days "d", months "m", year "y")
  - rebal_cost (bp)	= rebalancing cost
  - window = rolling window, in days, used to calculate the correlations between the pairs of the portfolio

You can find tests and analysis in the pdf "csmom_pairs_tests_and_analysis.pdf".

Note: the cross-sectional momentum strategy is considered the "benchmark" strategy.
