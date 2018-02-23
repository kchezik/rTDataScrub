library(tidyverse); library(lubridate); library(zoo); library(viridis); library(attenPlot); library(plotly); library(wmtsa); library(MSwM); library(attenPlot)

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
  data.frame(dat, wavDWT = dwt$data$d1)
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
set.seed(425)
site_samp = df %>% ungroup() %>% select(filename) %>% distinct() %>% sample_n(1)
test = df %>% filter(., filename %in% site_samp$filename)
ggplot(data = test, aes(DateTime, wavDWT, color = abs(wavDWT))) + geom_point() +
  facet_wrap(~filename) + scale_color_viridis()
ggplot(data = test, aes(DateTime, Dif, color = abs(wavDWT))) + geom_point() +
  facet_wrap(~filename) + scale_color_viridis()
ggplot(data = test, aes(DateTime, Tsd, color = abs(wavDWT))) + geom_point() +
  facet_wrap(~filename) + scale_color_viridis()
ggplot(data = test, aes(DateTime, Temperature, color = Dif)) + geom_point() +
  facet_wrap(~filename) + scale_color_viridis()

#Markov switching model.
library(MSwM)
wavMSwM=MSwM::msmFit(object = as.formula("wavDWT ~ 0"), data = test,
                      k=2, p=1, sw=c(F,T), control=list(parallel=F))

DifMSwM=MSwM::msmFit(object = as.formula("Dif ~ 0"), data = test,
                      k=2, p=1, sw=c(F,T), control=list(parallel=F))

TsdMSwM=MSwM::msmFit(object = as.formula("Tsd ~ 0"), data = test,
                      k=2, p=1, sw=c(F,T), control=list(parallel=F))

airD = which(DifMSwM@std == max(DifMSwM@std))
airW = which(wavMSwM@std == max(wavMSwM@std))
airS = which(TsdMSwM@std == max(TsdMSwM@std))

test = bind_cols(test, data.frame(air = zero_one(DifMSwM@Fit@smoProb[,airD] +
                                    wavMSwM@Fit@smoProb[,airW] +
                                    TsdMSwM@Fit@smoProb[,airS])))

#Look at how well air vs. water is being identified.
ggplot(test, aes(DateTime, Temperature, color = air)) + geom_point() + scale_color_viridis()


#New Variables

#Could simple use autocorrelation values instead of or in addition to difference values.


#Wavelet improvement.

#Retain the larger seasonal component and look for large deviations from the expectation. It may reveal the buried logger special case.


#Markov Switching Model Improvement

#Figure out how to give the MSwM a variance structure that changes over the season rather than simply switching between two static states. More verbosely we want two variance models that grow and decline on a sinusoidal curve resembling the season. In this way we can detect when we are in a different state or just when we are in a more variable part of the season but in a single state.

#It would also be great to compare these probabilities across sites and downweight or upweight depending on agreement.


## Special cases that need consideration. ##

#We need to test this on data where no air temperature was measured.
#Seed 425 may be when the logger is buried (10602068sept0615.csv).
