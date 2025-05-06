library(openair)
library(ggplot2)
library(prophet)
library(dplyr)

my1 = openair::importAURN(site = "MY1", year = 2017:2025)

df <- my1 %>% 
  rename(ds = date, y = no2)

m <- prophet(df)

future <- make_future_dataframe(m, periods = (730))

forecast <- predict(m, future)

plot(m, forecast) +
  ggtitle("NO2 Forecast for the Next 2 Years")

prophet_plot_components(m, forecast)


