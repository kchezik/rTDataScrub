library(tidyverse); library(lubridate)
setwd("~/sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Original_Data/HOBOware CSV/")
#Read in all data and parse into sites and logger ID.
df_T = data.frame()
for(i in dir()){
  parts = unlist(strsplit(i,"_"))
  dat = read_csv(file = i, skip = 2, col_names = c("row","date","temperature")) %>% select(-row) %>%
    mutate(site = parts[1], logger = unlist(strsplit(parts[3],split = ".", fixed = T))[1])
  df_T = bind_rows(df_T,dat)
}
#Change Column Class
df_T$date = ymd_hms(df_T$date)
df_T$site = as.numeric(df_T$site)
df_T$logger = as.numeric(df_T$logger)

setwd("~/Documents/rTDataScrub/Data/")
write_rds(x = df_T, path = "./thompson_water.rds")

#Read in ClimateBC annual and seasonal air temperature estimates
annual = dir()[grep(dir(), pattern = "climateBC_\\d+-\\d+YT")]
season = dir()[grep(dir(), pattern = "climateBC_\\d+-\\d+ST")]
#Calculate the mean annual air temperature for each site and year.
alpha = read_csv(annual) %>% group_by(ID1, Year) %>% summarise(air_mean = mean(MAT))
#Calculate the annual range in temperature for each site and year by taking the summer mean maximum temperature and adding the absolute value of the winter mean minimum temperature and dividing by two.
A = read_csv(season) %>% group_by(ID1, Year) %>% summarise(air_A = (max(Tmax_sm) + abs(min(Tmin_wt)))/2)
#Bind air temperature statistics into single database.
df_C = left_join(alpha,A) %>% rename(site = ID1, year = Year)
#Export summary.
write_rds(x = df_C, path = "./thompson_air_inits.rds")
