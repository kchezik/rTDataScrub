library(caret)

ggplot(test, aes(DateTime, Temperature, color = SD)) + geom_point() +
  facet_wrap(~filename) + scale_color_viridis()
ggplot(test, aes(Dif, SD, color = wavDWT)) + geom_point() +
  scale_color_viridis()

#Create two dataframes, the original and a dummy one sampled from the original.
class1 = test %>% ungroup() %>% select(Temperature, Dif, SD) %>% mutate(Class = 1)
class2 = data_frame(Temperature = sample(x = class1$Temperature, size = nrow(class1), replace = T),
                    Dif = sample(x = class1$Dif, size = nrow(class1), replace = T),
                    SD = sample(x = class1$SD, size = nrow(class1), replace = T)) %>%
  mutate(Class = 2)

df_fit = bind_rows(class1, class2)
df_fit$Class = as.factor(df_fit$Class)

trainIndex = caret::createDataPartition(df_fit$Class, p = .8,
                                  list = FALSE,
                                  times = 1)

training <- df_fit[trainIndex,]
testing  <- df_fit[-trainIndex,]

fitControl <- caret::trainControl(
  ## 10-fold CV
  method = "repeatedcv",
  number = 10,
  ## repeated ten times
  repeats = 10)

rfFit <- caret::train(Class ~ ., data = training,
                 method = "rf",
                 trControl = fitControl,
                 ## This last option is actually one
                 ## for gbm() that passes through
                 verbose = FALSE)
rfFit

preds = predict(rfFit, newdata = testing, type = "prob")
preds = bind_cols(testing, preds)
ggplot(preds, aes(`1`, `2`, color = Class, alpha = 0.1)) + geom_jitter(width = 0.2, height = 0.2)
write_rds(x = rfFit, path = "rfFit.rds")
