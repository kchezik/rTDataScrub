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
  ggplot(., aes(doy, t_val, col = source, alpha = 0.5)) + geom_point() +
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
  ggplot(., aes(doy, t_val, col = source, alpha = 0.5)) + geom_point() +
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
