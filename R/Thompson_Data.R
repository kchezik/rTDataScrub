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

files = dir()[grep(dir(), pattern = "climateBC*")]

df_C = data.frame()
for(i in files){
  dat = read_csv(i) %>% select(ID1, MAT, TD) %>% distinct() %>%
    group_by(ID1) %>% summarize(MAT = mean(MAT), TD = mean(TD)) %>%
    mutate(year = as.numeric(stringr::str_extract(i, pattern = "\\d+")))
  df_C = bind_rows(df_C, dat)
}

ggplot(df_C, aes(ID1, MAT, color = as.character(year))) + geom_point()
ggplot(df_C, aes(ID1, TD, color = as.character(year))) + geom_point()

df_C = df_C %>% rename(site = ID1, air_mean = MAT, air_A = TD, year = year)

write_rds(x = df_C, path = "./thompson_air_inits.rds")
