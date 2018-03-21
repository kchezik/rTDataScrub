#Get necessary libraries.
library(tidyverse);library(rstan)
#Initiate cores.
options(mc.cores = parallel::detectCores())

#Read in raw example dataset.
test = read_rds(path = "./Data/test.rds") %>% mutate(date = lubridate::round_date(DateTime, unit = "day"),
                                                     day = lubridate::yday(DateTime))
test$season = sin(scales::rescale(test$day, to = c(-pi/2, pi+(pi/2)), from = c(1,366)))+2
#Summarize hourly data up to the daily level.
d_test = test %>%
  mutate(date = lubridate::round_date(DateTime, unit = "day")) %>%
  group_by(date) %>% summarise(temperature = mean(Temperature, na.rm = T),
                               season = mean(season)) %>%
  mutate(lt = log((temperature + abs(min(temperature)))+1),
                     diff = c(NA, diff(temperature)),
                     ldiff = c(NA, diff(lt)))
#Fill in NA values
d_test[is.na(d_test$diff),"diff"] = 0
d_test[is.na(d_test$ldiff),"ldiff"] = 0

#Compile and run model.
compile <- stan_model("./stan/AR1_RW_RW0_Daily.stan")
mod<- sampling(compile, data= list(N = nrow(d_test), y = d_test$temperature, season = d_test$season),
                warmup = 1000, iter = 1500, chains = 4)
#Print random variable estimates.
print(mod, pars = c("p","alpha", "rho", "sigma"))

#View
est <- as.data.frame(mod, pars = "xi") %>%
  gather(par, value) %>%
  filter(grepl(",1", par)) %>%
  mutate(time = readr::parse_number(stringr::str_extract(par, "[0-9]{1,4}[,]"))) %>%
  group_by(par) %>%
  summarise(time = first(time),
            mean = mean(value),
            lower = quantile(value, .025),
            upper = quantile(value, .975)) %>%
  arrange(time) %>%
  mutate(date = d_test$date,
         temperature = d_test$temperature,
         diff = d_test$ldiff)

p1 = ggplot(est, aes(x = date)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), fill= "orange", alpha = 0.4) +
  geom_line(aes(y = mean)) +
  ggthemes::theme_economist() +
  xlab("Date") +
  ylab("Probability of air temperature")

temp = est %>% dplyr::select(date, mean) %>% left_join(test)
#temp = dat %>% filter(source == "test") %>% mutate(Date = lubridate::round_date(date, unit = "day"))
#temp = est %>% select(date, mean) %>% left_join(temp,., by = c("Date"="date"))

p2 = ggplot(temp, aes(x = DateTime, y = Temperature, color = mean)) +
  geom_point() +
  xlab("Date") +
  ylab("Temperature") + viridis::scale_color_viridis(limits = c(0,1)) +
  ggthemes::theme_economist() +
  theme(legend.position = "none")

gridExtra::grid.arrange(p1, p2, p3)



#Time varying finite mixture model for comparison.
mod1 = read_rds("stan/Mod_Results/MSwM.rds")
print(mod1, pars = c("beta","rho","sigma"))
est = mod1 %>%
  as.data.frame() %>%
  select(contains("mu")) %>%
  reshape2::melt() %>%
  group_by(variable) %>%
  summarise(lower = quantile(value, 0.95),
            median = median(value),
            upper = quantile(value, 0.05)) %>%
  mutate(date = test$DateTime,
         ac = test$Temperature)

p3 = ggplot(est, aes(date, ac, color = arm::invlogit(median))) + geom_point() +
  xlab("Date") +
  ylab("Temperature") + viridis::scale_color_viridis(direction = -1) +
  ggthemes::theme_economist() +
  theme(legend.position = "none")


full_mat = outer(test$Dif, test$Dif, '-')
plot(full_mat)

