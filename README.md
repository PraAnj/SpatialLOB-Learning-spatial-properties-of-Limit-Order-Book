# SpatialLOB-Learning-spatial-properties-of-Limit-Order-Book

SpatialLOB is designed for stock price prediction by exploiting spatial properties of the Limit Order book.

Orderbook micro structure consists of multiple price level on each side of the book, as buy side and sell side. These price levels could have different aggregate order volumes. Order volumes of a price points are changing as the price point receives order updates like add, amend, cancel orders. Following figure shows this spatial changes over multiple orderbook updates. Please refer [kaggle notebook](https://www.kaggle.com/praanj/limit-orderbook-visualizer-plotly) for more understanding of this view.

<img src="https://github.com/PraAnj/SpatialLOB-Learning-spatial-properties-of-Limit-Order-Book/blob/main/utility/orderbook_spatial_view.gif" width="600"/>
