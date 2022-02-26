
sigmoid  = function(z) 1 / (1 + exp(-z))

MSE = function(x,y) mean((x-y)^2)

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


train_nn = function(mod,x,y,loss,epochs = 20,learning_rate=.001,batch_size=nrow(x),metrics=NULL) {
  
  #batch_size = 32 test
  
  
  if(is.data.frame(x)) x = as.matrix(x)
  if(is.data.frame(y)) y = as.matrix(y)
  
  
  mod %>% compile(
    loss = loss,
    optimizer = optimizer_adam(learning_rate),
    metrics = metrics
  )
  
  
  history = mod %>% fit(
    x,
    y,
    epochs = epochs,
    batch_size = batch_size,
    verbose=1
  )
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




#' #' Title
#' #'
#' #' @param object 
#' #' @param newdata 
#' #' @param type 
#' #' @param se.fit 
#' #' @param dispersion 
#' #' @param terms 
#' #' @param na.action 
#' #' @param ... 
#' #'
#' #' @return
#' #' @export
#' #'
#' #' @examples
#' predict.glm = function(object, newdata = NULL,
#'                        type = c("response"),
#'                        se.fit = FALSE, dispersion = NULL, terms = NULL,
#'                        na.action = na.pass, ...) {
#'   
#'   stats::predict.glm(object=object, newdata = newdata,
#'                      type = type,
#'                      se.fit = se.fit, dispersion = dispersion, terms = terms,
#'                      na.action = na.action, ...)
#' }
  



