
library(Hmisc);library(raster);library(sp);library(rgdal); library(tidyverse);library(lubridate)
setwd("/Users/kylechezik/sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Original_Data/DFO Temperature Data")
loc = mdb.get("Store.mdb", tables="zGeoLocations") #All sites in original db.
setwd("~/sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Subset Data")
locLim = read_rds("01_Site_Locality_Data.rds") #Fraser locations in cleaned temperature data.
setwd("~/Documents/rTDataScrub")
df = read_rds("LabTemp_Data.rds")

#Visual inspection functions.
viewSite = function(dat, sites, view = F){
  if(view == T){
    loc %>% filter(GeoLocID%in%sites)
  }
}

viewData = function(tDF,sites, view = F){
  if(view == T){
    tDF %>% filter(geolocid%in%sites) %>%
      ggplot(.,aes(date,temperature,colour=as.factor(geolocid))) + geom_point()
  }
}

newID = function(df,sites,no){
  df[which(df$geolocid %in% sites),"newid"] <<- as.integer(no)
}

#Inspect
sites = c(36,39)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,36)

#Inspect
sites = c(178,209,1115)
viewSite(loc, sites)
viewData(df,sites)
#Remove 1115 as they are lake data. Combine 209 and 178
df = df %>% filter(geolocid!=1115)
newID(df,sites,178)

#Inspect
sites = c(1666,1668)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,1666)

#Inspect
sites = c(1584,1667)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,1584)

#Inspect
sites = c(1619,1621)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,1621)

#Inspect
sites = c(1128,31)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,1128)

#Inspect
sites = c(135,1138)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1138)

#Inspect
sites = c(216, 1153)
viewSite(loc, sites)
viewData(df,sites)
#Combine
newID(df,sites,1153)

#Inspect
sites = c(207, 1141)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1141)

#Inspect
sites = c(1589, 1605)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1605)

#Inspect
sites = c(110, 1569)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,110)

#Inspect
sites = c(1615, 1616)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1615)

#Inspect
sites = c(175, 1614)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,175)

#Inspect
sites = c(73,74)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,74)

#Inspect
sites = c(1576,1629)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1576)

#Inspect
sites = c(87,1654)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1654)

#Inspect
sites = c(1652,1653)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1652)

#Inspect
sites = c(112,1606)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1606)

#Inspect
sites = c(1655,1656)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1655)

#Inspect
sites = c(1633,1634)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1633)

#Inspect
sites = c(1643,1647,1650)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1643)

#Inspect
sites = c(1640,1641)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1640)

#Inspect
sites = c(1638,1639)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1639)

#Inspect
sites = c(1635,1646)
viewSite(loc, sites)
viewData(df, sites)
#Combine
newID(df,sites,1646)

#Clean up the combined data by averaging coincident data.
df$newid = if_else(is.na(df$newid), df$geolocid, as.integer(df$newid))
final = df %>% group_by(measurement, newid, date, error_lab) %>%
  summarise(temperature = mean(temperature)) %>%
  select(newid, measurement, date, error_lab, temperature)
#Join location data from the original database.
final = loc %>% select(GeoLocID, Latitude, Longitude, NAD, UTM.Zone, UTM.EastCoordinate, UTM.NorthCoordinate) %>% left_join(final, ., by = c("newid"="GeoLocID"))
#Join location data from the personally organized Fraser Location data.
final = locLim %>% select(GeoLocID, Zone, Easting, Northing, Latitude, Longitude) %>%  left_join(final, ., by = c("newid"="GeoLocID"))
#Compare NA values in and share data between location datasets.
final[is.na(final$Latitude.y) & !is.na(final$Latitude.x),"Latitude.y"][[1]]
final[is.na(final$Longitude.y) & !is.na(final$Longitude.x),"Longitude.y"][[1]]
final[is.na(final$Northing) & !is.na(final$UTM.NorthCoordinate),"Zone"][[1]] = as.character(final[is.na(final$Northing) & !is.na(final$UTM.NorthCoordinate),"UTM.Zone"][[1]])
final[is.na(final$Easting) & !is.na(final$UTM.EastCoordinate),"Easting"][[1]] = final[is.na(final$Easting) & !is.na(final$UTM.EastCoordinate),"UTM.EastCoordinate"][[1]]
final[is.na(final$Northing) & !is.na(final$UTM.NorthCoordinate),"Northing"][[1]] = final[is.na(final$Northing) & !is.na(final$UTM.NorthCoordinate),"UTM.NorthCoordinate"][[1]]
#Remove unused or now duplicate columns.
final = final %>% select(-UTM.Zone, -UTM.EastCoordinate, -UTM.NorthCoordinate, -Latitude.x, -Longitude.x, -NAD)
names(final)[9] = "Latitude"; names(final)[10] = "Longitude"


#### Standardize the Coordinate Reference System to lat/long in WGS84 from original Datum/CRS ####
WGS84 = crs("+init=epsg:4326"); BCAlbers = crs("+init=epsg:3005")
UTM10N = crs("+init=epsg:3157"); UTM11N = crs("+init=epsg:2955")

GeoL = final %>% ungroup() %>% select(1, 6:10) %>% distinct() %>% group_by(Zone) %>% do({
  if(all(.["Zone"]=="BC.Albers")){
    coords = .[,c("Easting","Northing")]
    sp.BC = SpatialPoints(coords = coords, proj4string = BCAlbers)
    sp.WGS = spTransform(x = sp.BC, CRSobj = WGS84)
    dat = data.frame(newid = .$newid, Zone = "BC.Albers", Easting = coordinates(sp.BC)[,"Easting"], Northing = coordinates(sp.BC)[,"Northing"], Latitude = coordinates(sp.WGS)[,"Northing"], Longitude = coordinates(sp.WGS)[,"Easting"])
  }
  if(all(.["Zone"]=="10N" | .["Zone"]=="10U")){
    coords = .[,c("Easting","Northing")]
    sp = SpatialPoints(coords = coords, proj4string = UTM10N)
    sp.BC = spTransform(x = sp, CRSobj = BCAlbers)
    sp.WGS = spTransform(x = sp, CRSobj = WGS84)
    dat = data.frame(newid = .$newid, Zone = "BC.Albers", Easting = coordinates(sp.BC)[,"Easting"], Northing = coordinates(sp.BC)[,"Northing"], Latitude = coordinates(sp.WGS)[,"Northing"], Longitude = coordinates(sp.WGS)[,"Easting"])
  }
  bind_cols = dat
})
#Add complete location data to the dataset.
final = final %>% ungroup() %>% select(-c(6:10)) %>%left_join(.,GeoL, by = "newid")

#Make GeoL a spatial object.
coords = GeoL[,c("Easting","Northing")]
sp.BC = SpatialPoints(coords = coords, proj4string = BCAlbers)
#Extract elevation data from the clipped grid dataset.
dem = raster::raster(x = "~/sfuvault/Simon_Fraser_University/PhD_Research/Projects/Data/Original_Data/GIS_Data/DEMs/BC_DEM/dem_merge/mergeDEM.tif", band = 1)
sp.BC$elevation = raster::extract(x = dem, y = sp.BC, method = 'simple')
sp.BC = data.frame(sp.BC) %>% select(-optional)
#Add elevation to the temperature dataframe.
final = final %>% left_join(., sp.BC, by = c("Northing","Easting"))
write_rds(final, "LabTempLoc_Data.rds")
