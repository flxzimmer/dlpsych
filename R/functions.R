


#' Title
#'
#' @param z 
#'
#' @return
#' @export
#'
#' @examples
sigmoid  = function(z) {1 / (1 + exp(-z))}

#

#' Title
#'
#' @param x 
#' @param y 
#'
#' @return
#' @export
#'
#' @examples
MSE = function(x,y) {mean((x-y)^2)}



#' Title
#'
#' @param x 
#' @param y 
#'
#' @return
#' @export
#'
#' @examples
CE = function(x,y) {
  -mean(x *log(y) + (1-x) *log(1- y))
  }



#' Title
#'
#' @param sample_size 
#'
#' @return
#' @export
#'
#' @examples
sim_data = function(sample_size) {
  
  agemean= 30
  agesd = 10
  cutoff = 18
  age = runif(sample_size,pnorm(cutoff,agemean,agesd),1)
  age = round(qnorm(age,agemean,agesd)) # 18-open end
  
  edu = 1+rbinom(sample_size, 4,sigmoid(scale(age)/3)) # 1-5 (Hauptschule -> Uni)
  
  polori = 1+rbinom(sample_size, 8,sigmoid(scale(age)/10)) # 1-9 (links -> rechts)
  cm = 5+rbinom(sample_size, 20,sigmoid(scale(age)/10+scale((scale(polori)+.4)^2)+scale(polori)/6-scale(edu)/9)) #5 items from likert scale
  
  df = data.frame(age,edu,polori,cm)
  return(df)
}



#' Keras Modelle trainieren
#' 
#' One-Line Interface to keras
#'
#' @param mod Keras Model, z.B. erstellt mit keras_model_sequential() 
#' @param x Prädiktoren
#' @param y Kriterium 
#' @param loss Loss Function, z.B. "mse" (Mean-Squared-Error) oder "ce" (cross-entropy) 
#' @param epochs Anzahl der Schritte im Gradient Descent. Default ist 20.
#' @param learning_rate Schrittweite im Gradient Descent. Default ist .001.
#' @param batch_size Für Stochastic Gradient Descent: Anzahl der Samples in einem Batch 
#' @param metrics (Optional) Weitere Metriken die neben der Loss beim Training angezeigt werden sollen. Diese werden beim Training nicht berücksichtigt. 
#' @param optimizer Verwendeter Optimizer, Default ist "adam". Alternativ: "rmsprop"
#' @param silent Wenn TRUE nur finalen Loss in der Console. Default ist FALSE
#'
#' @return
#' @export
#'
#' @examples
train_nn = function(mod,x,y,loss,epochs = 20,learning_rate=.001,optimizer="adam",batch_size=nrow(x),metrics=NULL,silent=FALSE) {
  
  if(is.data.frame(x)) x = as.matrix(x)
  if(is.data.frame(y)) y = as.matrix(y)
  
  if(tolower(optimizer)=="adam") optim = optimizer_adam(learning_rate)
  if(tolower(optimizer)=="rmsprop") optim = optimizer_rmsprop(learning_rate)
  
  if (tolower(loss)=="ce") loss = "binary_crossentropy"
  
  mod %>% compile(
    loss = loss,
    optimizer = optim,
    metrics = metrics
  )
  
  verbosity =  ifelse(silent, 0, 1) 
  
  history = mod %>% fit(
    x,
    y,
    epochs = epochs,
    batch_size = batch_size,
    verbose=verbosity
  )
  
  if(verbosity==0)   {
    
    final_loss = history$metrics$loss[length(history$metrics$loss)]
  print(paste0("Final Loss (",loss,"): ",final_loss))
  
  }
  
  return(mod)
}




plot_img = function(img1) {
  
  dat = c()
  for (i in 1:nrow(img1)) {
    for (j in 1:ncol(img1)) {
      dat = rbind(dat,c(i,j,img1[i,j]))
    }
  }
  dat = as.data.frame(dat)
  names(dat) = c("X","Y","val")
  
  ggplot() + geom_raster(data =dat,aes(x = Y, y = -X,fill=val)) + 
    scale_fill_gradient(low="grey90", high="black") + theme(legend.position="none")
}


#' Title
#'
#' @param n 
#' @param binary 
#'
#' @return
#' @export
#'
#' @examples
data_income = function(n,binary=FALSE) {
  
  education = rnorm(n)
  
  err_sd = 2
  income_sd = 1.3
  income_mean = 6.5
  income = 1 * education + rnorm(n,sd=err_sd)
  income = scale(income)*income_sd+ income_mean
  
  n_unrelated = 15
  unrelated = matrix(rnorm(n*n_unrelated),ncol=n_unrelated)
  unrelated = as.data.frame(unrelated)
  names(unrelated) = paste0("unrelated",1:n_unrelated)
# n=100000
# mean(income)
# sd(income)
# cor(education,income)

  dat = data.frame(income=income,education=education)
  dat = cbind(dat,unrelated)
  
  if (binary) {
    dat$income = as.numeric(dat$income>median(dat$income))
    # dat$income = ifelse((dat$income>median(dat$income)),"high","low")
    # dat$income = factor(dat$income,levels = c("low","high"))
  }
    
  return(dat)
}


#' Title
#'
#' @param n 
#' @param binary 
#'
#' @return
#' @export
#'
#' @examples
data_income_challenge = function(type,binary=FALSE) {

  # train data 
  n=200
  
  dat = data_income(n=n,binary=binary)
  
  dat$income= dat$income *3
  
  dat$unrelated1 = dat$unrelated1 *10
  
  if(type=="train") return(dat)
  
  # test data 
  n=50
  
  dat = data_income(n=n,binary=binary)
  
  dat$income= dat$income *3
  
  dat$unrelated1 = dat$unrelated1 *10
  
  return(dat)
}


#' Title
#'
#' @param mod_nn 
#'
#' @return
#' @export
#'
#' @examples
weights_nn = function(mod_nn) {
  
  a = get_weights(mod_nn)
  
  len = length(a)
  
  ind_weights = seq(1,len,by=2)  
  ind_bias = seq(2,len,by=2)  
  
  names(a) = seq(1,len)
  names(a)[ind_weights] = paste0("Weights_Layer_",1:(len/2))
  names(a)[ind_bias] = paste0("Bias_Layer_",1:(len/2))
  
  for (i in ind_weights) {
    temp = a[[i]]
    rownames(temp) = paste0("pred",1:nrow(temp))
    colnames(temp) = paste0("neuron",1:ncol(temp))
    a[[i]] = temp
  }
  
  return(a)
  
}


