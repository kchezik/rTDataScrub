library(tidyverse); library(lubridate)
############### Clean Duplicate Free Data ###############
#hourly
df_T = read_rds("./Data/thompson_water.rds") %>%
  filter(site %in% c(49,63,54,39,1,80,87,93,100,77)) %>%
  mutate(error_lab = NA, row = row_number())
#daily
#df = df_T %>% mutate(date = floor_date(date, unit = "day")) %>%
#  group_by(site,date) %>% summarise(temperature = mean(temperature)) %>% ungroup() %>%
#  mutate(error_lab = NA, row = row_number())

############### Functions for Going Through Sites Point by Point ###############
#Print the Data in Question and Plot the Data.
PrintPlotOpts = function(subDat, fullDat){
	print(subDat) #Print data in question to screen.
  current = ymd(floor_date(subDat$date, unit="day"))
	initialDate = current-150; finalDate = current+150 #Create data window for viewing
	temp = fullDat %>% filter(site == unique(subDat$site),date>=initialDate, date<=finalDate)
	p = ggplot(temp, aes(date,temperature)) + geom_point(alpha = 0.5) +
	  geom_point(data = subDat, aes(date, temperature), color = "red")#Create plot for inspection.
	print(p) #Print plot.
}
#Ask how you would like to label the data currently in view.
SelectError = function(all, ans, no){
	if(all == F){
		repeat{
			ans = readline("What is the apparent error? (1:air, 2:water, 3:error_unk, 4:choose error for all remaining., 5:error over # days.) ")
			if(ans==1 | ans==2 | ans==3 | ans==4 | ans==5) break
		}
	}
	if(all == T & is.null(ans)){
		repeat{
			ans = readline("What is the apparent error? (1:air, 2:water, 3:error_unk) ")
			if(ans==1 | ans==2 | ans==3) break
		}
	}
	if(ans == 4 | ans == 5) return(ans)
	else {
		if(ans == 1) return("air")
		if(ans == 2) return("water")
		if(ans == 3) return("error_unk")
	}
}
#Wrapper for plotting and cleaning.
clean.temp = function(df, dFull, all = F, ans = NULL, no = 0){
	error = df %>% group_by(date) %>%
		do({
			if(no == 0){
				if(all == F & no == 0){
					PrintPlotOpts(., dFull)
				}
				error_lab = SelectError(all, ans, no)
				if(error_lab == 4) {
					all = T
					error_lab = SelectError(all, ans, no)
					if(error_lab == "air") ans = 1
					if(error_lab == "water") ans = 2
					if(error_lab == "error_unk") ans = 3
				}
				if(all == T & is.null(ans)==F){
					if(ans == 1) error_lab = "air"
					if(ans == 2) error_lab = "water"
					if(ans == 3) error_lab = "error_unk"
				}
				if(error_lab == 5) {
					no = as.numeric(readline("How many days? "))
					error_lab = SelectError(all = T, ans, no)
					if(error_lab == "air") ans = 1
					if(error_lab == "water") ans = 2
					if(error_lab == "error_unk") ans = 3
				}
			}
			if(all == F & no == 0){
				dev.off()
			}
			if(no != 0){
				if(ans == 1) error_lab = "air"
				if(ans == 2) error_lab = "water"
				if(ans == 3) error_lab = "error_unk"
				no = as.numeric(no) - 1
				if(no == 0) ans = NULL
			}
			data.frame(error_lab, row = .$row)
		})
	error
}
#Function for taking a specific slice of the data in time.
narrow = function(df, s, startDate, endDate){
	temp = df %>% filter(site==s, date >= startDate, date <= endDate)
	return(temp)
}
#Add appropriate labels to the data given the result from `clean.temp`.
add = function(df, rtrn){
	df_T[rtrn$row,"error_lab"] <<- as.character(rtrn$error_lab)
}
#Produce objects indicating the full range of dates in the data.
full_dates = function(df, s, start = F, end = F){
	if(start == T) return(df %>% filter(site==s) %>% .$date %>% min())
	if(end == T) return(df %>% filter(site==s) %>% .$date %>%  max())
}
#Produce a date quickly for taking a slice of the data.
iso_date = function(year, month, day){
	year = as.character(year); month = as.character(month); day = as.character(day)
	return(ymd(paste(year,"-",month,"-",day, sep="")))
}
#Write the result of `clean.temp` to a hard .rds file.
WriteRead = function(obj, year, s, write = T){
	if(write == T){
		write_rds(x = obj, path = paste("./Data/direct_cleaning/Y",year,"_",as.character(s),".rds", sep = ""))
	}
	if(write == F){
		read_rds(paste("./Data/direct_cleaning/Y",year,"_",as.character(s),".rds", sep = ""))
	}
}
getPlot = function(df, screen = F){
	if(screen == T){
		t = paste0("Site:",
					 as.character(unique(df$site)),
					 " Year:",
					 as.character(unique(year(df$date))))
		if(sum(is.na(df$error_lab)) == nrow(df)) {
		  ggplot(df, aes(date, temperature)) + geom_point()	+ labs(title = t)
		} else {
		  ggplot(df, aes(date, temperature, color = error_lab)) + geom_point()	+ labs(title = t)
		}
	}
}

##########################################################################################

#Site/Year combinations.
Site = df %>% ungroup() %>% select(site) %>% distinct() %>% .$site

#Hourly
Site = df_T %>% ungroup() %>% filter(site %in% c(49,63,54,39,1,80,87,93,100,77)) %>%
  select(site) %>% distinct() %>% .$site

###################Site 1######################
s = Site[1]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,15))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
#rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,04), iso_date(year,10,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,09,03), iso_date(year,09,06))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,21), iso_date(year,09,23))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,07,27), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 39#####################
s = Site[2]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,15))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
#rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,07), iso_date(year,10,10))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,07,17), iso_date(year,09,11))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,01,01), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 49#####################
s = Site[3]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,22))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,08,23), iso_date(year,09,15))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,09,15), iso_date(year,10,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

#df_T[(df_T$site == 49 & df_T$logger==10504132 & df_T$date>ymd_h("2014-10-10 14")),"error_lab"] = "air"
#df_T[(df_T$site == 49 & df_T$logger==10602108 & df_T$date>ymd_h("2014-10-04 18") & df_T$date<ymd_h("2014-10-09 21")),"error_lab"] = "air"

year = "2015"
period = narrow(df_T, s, iso_date(year,12,21), iso_date(year,12,31))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,26), iso_date(year,09,28))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,08,18), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 54#####################
s = Site[4]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,22))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,10), iso_date(year,10,12))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,09,08), iso_date(year,12,31))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,25), iso_date(year,09,27))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,08,18), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################

###################Site 63#####################
s = Site[5]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,10), iso_date(year,10,12))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,08,12), iso_date(year,09,12))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,25), iso_date(year,09,27))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,08,18), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 77#####################
s = Site[6]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,12), iso_date(year,10,14))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,09,11), iso_date(year,09,13))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,08,18), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 80#####################
s = Site[7]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,13), iso_date(year,10,15))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,09,12), iso_date(year,09,16))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,20), iso_date(year,09,22))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

#df_T[df_T$site==80 & df_T$logger==10602065 & df_T$date>ymd_h("2016-09-20 10"),"error_lab"] = "air"

year = "2017"
period = narrow(df_T, s, iso_date(year,06,25), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 87#####################
s = Site[8]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,08,07))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,08,10), iso_date(year,10,01))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2014"
period = narrow(df_T, s, iso_date(year,10,01), iso_date(year,10,15))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
#df_T[89609,"error_lab"] = "air"

year = "2015"
period = narrow(df_T, s, iso_date(year,09,11), iso_date(year,09,16))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2016"
period = narrow(df_T, s, iso_date(year,09,27), iso_date(year,09,29))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)


year = "2017"
period = narrow(df_T, s, iso_date(year,08,19), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################

###################Site 93#####################
s = Site[9]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,10,01))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2015"
period = narrow(df_T, s, iso_date(year,09,01), iso_date(year,12,31))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

year = "2017"
period = narrow(df_T, s, iso_date(year,08,19), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


###################Site 100####################
s = Site[10]
## Full Width Dates ##
startDate = full_dates(df_T,s,T,F)
endDate = full_dates(df_T,s,F,T)

## View Data ##
temp = narrow(df_T, s, startDate, endDate)
getPlot(temp, screen = T)

## Label Errors ##
year = "2014"
period = narrow(df_T, s, startDate, iso_date(year,10,17))
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)

df_T[df_T$site==s & df_T$logger==10602147 & df_T$date<=ymd_hms("2015-09-03 07:17:47"),"error_lab"] = "air"
df_T[df_T$site==s & df_T$logger==10575374 & df_T$date>=ymd_hms("2015-09-03 07:17:47"),"error_lab"] = "air"

year = "2017"
period = narrow(df_T, s, iso_date(year,08,19), endDate)
#rtrn = clean.temp(period, df_T); WriteRead(rtrn, year, s, T)
rtrn = WriteRead(rtrn, year, s, F)
add(df_T, rtrn)
###############################################


#Complete
df_T[df_T$temperature<0,"error_lab"] = "air"
df_T[is.na(df_T$error_lab),"error_lab"] = "water"
write_rds(df_T, "./Data/thompson_hourly_water_manual_clean.rds")
