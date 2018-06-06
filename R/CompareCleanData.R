library(tidyverse)
#Water data is from the DFO database for GeoLocID 51 on the Fraser River 10km south of Lillooet. (2008)
#Air temperature data is from lillooet. (2008)

#Read in labeled data.
dat = read_rds(path = "./Data/test_labeled.rds")
#Read in test data and configure.
test = read_rds("./Data/test.rds") %>% ungroup() %>%select(DateTime, Temperature)
names(test) = c("date","temperature")
test$source = "test"
#Bind data together and add day of year column.
dat = bind_rows(dat, test) %>% mutate(doy = lubridate::yday(date))


#Add log transformed and differenced temperature.
dat = dat %>% filter(!is.na(temperature)) %>% arrange(source, date) %>%
  mutate(lt = log((temperature + abs(min(temperature)))+100)) %>% group_by(source) %>%
  mutate(diff = c(NA, diff(temperature)),
         ldiff = c(NA, diff(lt))) %>% ungroup()

#Gather temperature data into long format.
dat = dat %>% gather(key = "transform", value = "t_val", temperature, lt, diff, ldiff)

#Plot hourly data.
dat %>% arrange(source) %>%
  ggplot(., aes(doy, t_val, col = source, alpha = 0.5)) + geom_line() +
  facet_wrap(~transform, scales = "free")

#ACF plots
d_spread = dat %>% spread(., key = transform, value = t_val) %>% arrange(date)
par(mfrow = c(3,2))
d_spread %>% filter(source == "water") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "h_w_diff")
d_spread %>% filter(source == "water") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "h_w_ldiff")
d_spread %>% filter(source == "air") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "h_a_diff")
d_spread %>% filter(source == "air") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "h_a_ldiff")
d_spread %>% filter(source == "test") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "h_t_diff")
d_spread %>% filter(source == "test") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "h_t_ldiff")

#Create and look at daily data.
d_dat = d_spread %>% mutate(day = lubridate::round_date(date, unit = "day")) %>%
  group_by(source, day) %>% summarise(temperature = mean(temperature)) %>% ungroup() %>%
  mutate(lt = log((temperature + abs(min(temperature)))+100)) %>% group_by(source) %>%
  mutate(diff = c(NA, diff(temperature)),
         ldiff = c(NA, diff(lt))) %>% ungroup()

#Gather
d_dat = d_dat %>% gather(key = "transform", value = "t_val", temperature, lt, diff, ldiff) %>%
  mutate(doy = lubridate::yday(day))

#Plot daily data.
d_dat %>% arrange(source) %>%
  ggplot(., aes(doy, t_val, col = source, alpha = 0.5)) + geom_line() +
  facet_wrap(~transform, scales = "free")

#ACF plots
d_spread = d_dat %>% spread(., key = transform, value = t_val) %>% arrange(day)
par(mfrow = c(3,2))
d_spread %>% filter(source == "water") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "d_w_diff")
d_spread %>% filter(source == "water") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "d_w_ldiff")
d_spread %>% filter(source == "air") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "d_a_diff")
d_spread %>% filter(source == "air") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "d_a_ldiff")
d_spread %>% filter(source == "test") %>% .$diff %>% acf(x = ., na.action = na.pass, main = "d_t_diff")
d_spread %>% filter(source == "test") %>% .$ldiff %>% acf(x = ., na.action = na.pass, main = "d_t_ldiff")


# In comparing known air and water temperature data it appears that water temperature is more likely to have momentum than air. In other words after differencing the logged daily data at lag 1, the acf function shows no evidence of autocorrelation in the air but some evidence in water at lag 1 and 2. This suggests one type of data has an AR1 process while the other is just a random walk. In the hourly data, both data types show autocorrelation. As such, we may be able to tease apart the two processes by using a model with just random noise and the other having an AR1 process (see AR1_vs_RW_Daily.stan file).


x = d_dat %>% arrange(day) %>% filter(source == "air", transform == "temperature") %>% .$t_val
y = d_dat %>% arrange(day) %>% filter(source == "water", transform == "temperature") %>% .$t_val

aic_mat = matrix(nrow = 90, ncol = 4)
for(i in 10:100){
  print(i)

  ARmod = arima(x[1:i], order = c(1,0,0))
  RWmod = arima(x[1:i], order = c(0,0,0))

  aic_mat[i,1] = ARmod$aic
  aic_mat[i,2] = RWmod$aic

  ARmod = arima(y[1:i], order = c(1,0,0))
  RWmod = arima(y[1:i], order = c(0,0,0))

  aic_mat[i,3] = ARmod$aic
  aic_mat[i,4] = RWmod$aic
}
#This plot shows that at about 20 data points an AR1 model overtakes a Random Walk model for air temperature. Thus if the dataset has air temperature at the tips and tails it will likely be best described by a RW model but if the logger is exposed for more than 20 continuous datapoints it's probably more likley an AR1 model.
plot(aic_mat[1:60,1]~aic_mat[1:60,2], xlab = "RW", ylab = "AR")
abline(1,1)


