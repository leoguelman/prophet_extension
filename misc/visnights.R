library(fpp2)
library(hts)
library(dplyr)
library(lubridate)
library(zoo)

data(visnights)

tourism.hts <- hts(visnights, characters = c(3, 5))
tourism.hts %>% aggts(levels=0:1) %>%
  autoplot(facet=TRUE) +
  xlab("Year") + ylab("millions") + ggtitle("Visitor nights")

date1 <- as.yearmon(time(visnights), "%b %Y")
Year <- format(date1, "%Y") ## Year with century
Month <- format(date1, "%m") ## numeric month

visnights_data <- data.frame(
  year = Year,
  month = Month,
  NSWMetro = visnights[,1],
  NSWNthCo = visnights[,2],
  NSWSthCo = visnights[,3],
  NSWSthIn = visnights[,4],
  NSWNthIn = visnights[,5]
)

write.csv(x=visnights_data, file="visnights_data.csv",
          row.names = FALSE)

