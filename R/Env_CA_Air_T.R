#Read in
library(tidyverse);library(rclimateca)

locations = read_csv("~/sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Original_Data/GIS_Data/Env_CA_Weather_Stn.csv")

loc = locations %>% .$STN_NAME %>% unique(.)

EC = data.frame()
for(j in loc){
  site = ec_climate_search_locations(j)
  for(i in site){
    df = ec_climate_data(location = i, timeframe = "daily",
                         start = "2014-07-01", end = "2017-10-01")
    if(nrow(df)>0){
      df = df %>% mutate(place = j)
      EC <<- df %>% select(2,3,4,5,6,8,10,12,30) %>% bind_rows(EC,.)
    }
  }
}

EC = EC %>% filter(location != "ARMSTRONG (AUT) ON 3987")
EC[grep(pattern = "^SUN",x = EC$place),"place"] = "SUN PEAKS"
EC[grep(pattern = "100|BUF|108", x = EC$place),"place"] = "100 MILE HOUSE"
EC[grep(pattern = "KAM", x = EC$place),"place"] = "KAMLOOPS"
EC[grep(pattern = "BLU", x = EC$place),"place"] = "BLUE RIVER"
EC[grep(pattern = "CLEAR", x = EC$place),"place"] = "CLEARWATER"
EC[grep(pattern = "SALM", x = EC$place),"place"] = "SALMON ARM"

ggplot(EC, aes(date, mean_temp_c, color = location)) +
  geom_line() +
  facet_wrap(~place)

EC %>% select(location) %>% distinct() %>% group_by(location) %>% .$location %>%
  ec_climate_search_locations(.) %>% tibble::as_tibble() %>%
  write_csv(., path = "~/Documents/rTDataScrub/Data/thompson_air_geo.csv")
write_rds(EC, "~/Documents/rTDataScrub/Data/thompson_air.rds")
