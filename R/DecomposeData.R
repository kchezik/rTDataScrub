library(tidyverse); library(lubridate); library(zoo); library(viridis); library(attenPlot); library(plotly); library(wmtsa); library(MSwM)

path = "../../sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Original_Data/HOBOware CSV/"
files = dir(path = path ,pattern = "*.csv")

df = data.frame(filename = files) %>%
  mutate(data = map(filename,
                    ~read_csv(file = file.path(path,.),
                              col_names = c("No","DateTime","Temperature"),
                              skip = 2))) %>%
  unnest()

#Make the date and time readable.
df$DateTime = mdy_hms(df$DateTime)

#This is a discrete wave transform that decomposes each time series into several frequencies depending on the length of the time period of interest. I am interested in only the coefficients describing the sub-daily cycles (D1) as they gives the clearest distinction between noisy air temperature periods and quiet water temperature periods.
df = df %>% group_by(filename) %>% do({
  dat = arrange(., DateTime) %>% filter(!is.na(Temperature))
  dwt = wavMODWT(x = dat$Temperature)
  wavD = dwt$data$d1
  #browser()
  #end = length(names(dwt$data)); end = ifelse(end>11, 11, end-1)
  #for(j in 1:end){
  #  for(i in 1:length(dwt$data[[j]])) dwt$data[[j]][i] = 0
  #}
  #wavS = reconstruct(dwt)
  #data.frame(dat[c(1:4)], wavD, wavS)
  data.frame(dat, wavD)
})

#Remove rows with missing temperature data and add differenced and rolling window sd data.
df = df %>% group_by(filename) %>%
  mutate(dif_r = c(NA,Temperature[1:(length(Temperature)-1)]-Temperature[2:length(Temperature)]),
         dif_l = c(Temperature[2:length(Temperature)]-Temperature[1:(length(Temperature)-1)], NA),
         tSD_l = rollapply(Temperature, width = 5, FUN = sd, fill = NA, align = "left"),
         tSD_r = rollapply(Temperature, width = 5, FUN = sd, fill = NA, align = "right"))

#Eliminate NA values by filling in left justified rolling average estimates ...
#... with right justified and vica versa. Fill right justifiied if on the right half of time series and fill left justified if on the left half of the time series.
df = df %>% mutate(
  Dif = if_else(is.na(dif_l), dif_r,
                if_else(DateTime<mean(DateTime), dif_l, dif_r)),
  Tsd = if_else(is.na(tSD_l), tSD_r,
               if_else(DateTime<mean(DateTime), tSD_l, tSD_r))) %>%
  select(-contains("_l"), -contains("_r"))

#Isolate just one site.
wegt = data.frame(weights = zero_one(sin(seq(-pi/2, pi+(pi/2), length.out = 366)))+1, day = c(1:366))
set.seed(425)
site_samp = df %>% ungroup() %>% select(filename) %>% distinct() %>% sample_n(1)
test = df %>% filter(., filename %in% site_samp$filename) %>%
  mutate(day = yday(DateTime)) %>% left_join(., wegt)
