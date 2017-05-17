#Podlaczenie bibliotek i czyszcenie konsoli
library(readr)
library(dplyr)
library(MASS)
library(dummies)
rm(list=ls())
cat("\014")

#Wczytywanie danych
train <- read_csv(paste(getwd(),"train.csv",sep="/"))
test <- read_csv(paste(getwd(),"test.csv",sep="/"))

#Stworzenie 'Dummy Variables' z zmiennych Departament Description i Weekday
train <- cbind.data.frame(train, dummy(train$DepartmentDescription), dummy(train$Weekday))
test <- cbind.data.frame(test, dummy(test$DepartmentDescription), dummy(test$Weekday))

#Usuniecie niepotrzebnych kolumn + HEALTH AND BEUTY AIDS nie pojawia się w test, co psuje obliczenia X.test%*% theta
test <- subset(test, select = -c(FinelineNumber, Weekday, DepartmentDescription, Upc))
train <- subset(train, select = -c(FinelineNumber, Weekday, DepartmentDescription, Upc, `trainHEALTH AND BEAUTY AIDS`))

#Utworzenie kolumny z TripType zgrupowanym przez VisitNumber
trip.type <- cbind.data.frame(train$TripType, train$VisitNumber)
trip.type <- aggregate(trip.type, list(trip.type$`train$VisitNumber`), FUN=mean)[2]

#Grupujemy po VisitNumber
train <- aggregate(train, list(train$VisitNumber), FUN=mean)
train <- train[c(2,4:ncol(train))]
test <- aggregate(test, list(test$VisitNumber), FUN=mean)
id <- test$VisitNumber
id <- as.integer(id)
test <- test[c(3:ncol(test))]

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

cost <- function(theta){
  m <- nrow(X)
  g <- sigmoid(X%*%theta)
  J <- 1/(2*m)*sum((-Y*log(g)) - ((1-Y)*log(1-g))) + (1/2/m)*(sum(theta^2)-theta[1]^2)
  return(J)
}

gradient <- function(theta){
  m <- nrow(X)
  grr <- 1/m*t(X)%*%(sigmoid(X%*%theta)-Y) + 1/m*(c(0, theta[-1]))
  return(grr)
}

#Przeksztalcamy df do macierzy, dla obliczeń 
X <- as.matrix(cbind.data.frame(1, train[2:ncol(train)]))
X.test <- as.matrix(cbind.data.frame(1,test))


#Normalizacja X oraz X.test
for(i in c(2:ncol(X))){
  m <- mean(X[,i])
  sd <- sd(X[,i])
  X[,i] <- (X[,i]-m)/sd
}

for(i in c(2:ncol(X.test))){
  m <- mean(X[,i])
  sd <- sd(X[,i])
  X.test[,i] <- (X.test[,i]-m)/sd
}

vec.theta <- data.frame()
#Tworzymy regresje logistyczną one vs rest
for(i in c(3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 19,
           20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
           42, 43, 44, 999)){
  Y <- trip.type
  Y[trip.type==i] <- 1
  Y[trip.type!=i] <- 0
  Y <- as.matrix(Y)
  initial.theta <- as.matrix(runif(ncol(X)))
  c <- cost(initial.theta)
  #Wybieramy wektor początkowy, dla którego cost jest sensowny(!= inf || != NaN)
  while(is.infinite(c) | is.na(c)){
    initial.theta <- as.matrix(runif(ncol(X)))
    c <- cost(initial.theta)
  }
  opt <- optim(par=initial.theta,fn=cost, gr=gradient)
  theta <- opt$par
  vec.theta <- rbind.data.frame(vec.theta, t(theta))
}

thetas <- t(as.matrix(vec.theta))
#Tworzymy predykcje dla danych testowych
predictions <- sigmoid(X.test%*%(thetas))
pred <- cbind.data.frame(as.integer(id), predictions)
colnames(pred) <- c("VisitNumber","TripType_3","TripType_4","TripType_5","TripType_6","TripType_7",
                    "TripType_8","TripType_9","TripType_12","TripType_14","TripType_15","TripType_18",
                    "TripType_19","TripType_20","TripType_21","TripType_22","TripType_23","TripType_24",
                    "TripType_25","TripType_26","TripType_27","TripType_28","TripType_29","TripType_30",
                    "TripType_31","TripType_32","TripType_33","TripType_34","TripType_35","TripType_36",
                    "TripType_37","TripType_38","TripType_39","TripType_40","TripType_41","TripType_42",
                    "TripType_43","TripType_44","TripType_999")
write.csv(pred, paste(getwd(),"pred.csv", sep = "/") , row.names=FALSE)