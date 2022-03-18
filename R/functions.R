

# helper ------------------------------------------------------------------

#' Title
#' 
#'
#' @param z 
#'
#' @return
#' @export
#'
#' @examples
sigmoid  = function(z) {1 / (1 + exp(-z))}


#' Title
#'
#' @param vec 
#'
#' @return
#' @export
#'
#' @examples
softmax = function(vec) {
  exp(vec) / sum(exp(vec))
}


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
#' @param y 
#' @param pred 
#'
#' @return
#' @export
#'
#' @examples
CE = function(y,pred) {
  
  if(length(unique(pred))==2) warning("wrong argument order? first argument should be the true values, second the prediction.")
  
  pred[pred == 0] = 1e-8
  pred[pred == 1] = 1-1e-8
  
  -mean(y *log(pred) + (1-y) *log(1- pred))
  }



#' Title
#'
#' @param y 
#' @param pred 
#'
#' @return
#' @export
#'
#' @examples
CCE = function(y,pred) {
  
  if(length(unique(pred))==2) warning("wrong argument order? first argument should be the true values, second the prediction.")
  
  # sparse cce
  if(length(dim(y)) == 1) {
    re = as.numeric(loss_sparse_categorical_crossentropy(y, pred))
  }
  
  # normal cce
  if(length(dim(y)) > 1) {
    re = as.numeric(loss_categorical_crossentropy(y, pred))
  }
  
  return(mean(re))
}



#' Title
#'
#' @param y 
#' @param pred 
#'
#' @return
#' @export
#'
#' @examples
accuracy = function(y,pred) {
  
  if(length(unique(pred))==2||is.integer(pred)) warning("wrong argument order? first argument should be the true values, second the prediction.")
  
  # sparse cce
  if(length(dim(y)) == 1) {
    re = mean(apply(pred,1,function(x) which(x==max(x))[1])-1 == y)
      }
  
  # normal cce
  if(length(dim(y)) > 1) {
    re = mean(apply(pred,1,function(x) which(x==max(x))[1]) == apply(y,1,function(x) which(x==1)))
  }
  
  return(re)
}





#' Title
#'
#' @param mod 
#' @param dat_train 
#' @param dat_test 
#'
#' @return
#' @export
#'
#' @examples
accuracy_by_class = function(mod,dat_train,dat_test) {
  
  # predictions 
  pred_train = predict(mod,dat_train$x)
  pred_test = predict(mod,dat_test$x)
  
  #train accuracy
  acc_train = c()
  categories = sort(unique(dat_train$y))
  for (i in categories) {
    acc_train = c(acc_train,accuracy(dat_train$y[dat_train$y==i],pred_train[dat_train$y==i,]))
  }
  
  #test accuracy
  acc_test = c()
  categories = sort(unique(dat_test$y))
  for (i in categories) {
    acc_test = c(acc_test,accuracy(dat_test$y[dat_test$y==i],pred_test[dat_test$y==i,]))
  }
  
  dat = rbind(acc_train,acc_test)
  re = as.data.frame(dat)
  re = round(re,3)
  rownames(re) = c("train","test")
  colnames(re) = categories
  
  return(re)
}




#' Title
#'
#' @param img1 
#' @param color 
#' @param inv 
#'
#' @return
#' @export
#'
#' @examples
plot_img = function(img,color=FALSE,inv=FALSE) {
  
  dat = c()
  for (i in 1:nrow(img)) {
    for (j in 1:ncol(img)) {
      dat = rbind(dat,c(i,j,img[i,j]))
    }
  }
  dat = as.data.frame(dat)
  names(dat) = c("X","Y","val")
  
  
  p1 = ggplot() + geom_raster(data =dat,aes(x = Y, y = -X,fill=val)) + 
    theme_minimal() +
    theme(panel.grid = element_blank(),legend.position="none",aspect.ratio = 1,axis.text.x=element_blank(),axis.text.y=element_blank()) +  xlab("") + ylab("")
  
  if(color)  {
    if(inv) p1 = p1 + scale_fill_viridis_c(direction=1)
    if(!inv)   p1 = p1 + scale_fill_viridis_c(direction=-1)
  }
  if(!color)  {
    if(inv) p1 = p1 + scale_fill_gradient(low="black", high="grey90")
    if(!inv)   p1 = p1 + scale_fill_gradient(low="grey90", high="black")
  }
  
  print(p1)
}


#' Title
#'
#' @param mod 
#' @param dat_train 
#' @param dat_test 
#'
#' @return
#' @export
#'
#' @examples
evaluate_mod = function(mod,dat_train,dat_test,measures) {
  
  # predictions 
  pred_train = predict(mod,dat_train$x)
  pred_test = predict(mod,dat_test$x)
  
  re = c()
  
  for (i in measures) {
    
    if (tolower(i)=="cce") meas = dlpsych::CCE
    if (tolower(i)=="acc") meas = dlpsych::accuracy
    if (tolower(i)=="mse") meas = dlpsych::MSE
    if (tolower(i)=="ce") meas = dlpsych::CE
    
  val_train = meas(dat_train$y,pred_train)
  val_test = meas(dat_test$y,pred_test)

  re = cbind(re,c(val_train,val_test))
  
  }
  
  re = as.data.frame(re)
  colnames(re) = measures
  rownames(re) = c("train","test")
  return(re)
}




# data --------------------------------------------------------------------


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
#' @param n 
#' @param binary 
#'
#' @return
#' @export
#'
#' @examples
data_interaction = function(n,include_binary=FALSE) {
  
  fare_dodgers = rnorm(n)
  inspectors = rnorm(n)
  
  y = (fare_dodgers * inspectors)
  y = y - min(y)
  y = y *10
  
  dat = data.frame(fare_dodgers=fare_dodgers,inspectors=inspectors,profit=y)
  
  
  if(include_binary) {
  fare_dodgers_binary  = (fare_dodgers>0) %>% ifelse("high","low") %>% factor(.,levels=c("low","high"))
  inspectors_binary  = (inspectors>0) %>% ifelse("high","low") %>% factor(.,levels=c("low","high"))
  dat = data.frame(fare_dodgers=fare_dodgers,inspectors=inspectors,y=y,fare_dodgers_binary=fare_dodgers_binary, inspectors_binary=inspectors_binary)
  
  dat = data.frame(fare_dodgers=fare_dodgers,inspectors=inspectors,profit=y,fare_dodgers_binary=fare_dodgers_binary, inspectors_binary=inspectors_binary)
  }
  
  return(dat)
}



#' From Urban,Gates 2021 https://doi.org/10.1037/met0000374
#'
#' @param n 
#'
#' @return
#' @export
#'
#' @examples
data_alcohol = function(n,extended =TRUE) {
  #extended includes more questionnaire items
  
  gender = as.numeric(rbinom(n, 1, 0.5))
  age = round(runif(n, 18, 85), 0); age = (age - min(age)) / (max(age) - min(age))
  mud = 1 / (1 + exp(1 + age - 2*gender)); mud = sapply(mud, function(prob) rbinom(1, 1, prob))
  nud = 1 / (1 + exp(1 + age - 2*gender - 0.5*mud)); nud = sapply(nud, function(prob) rbinom(1, 1, prob))
  mdd = 1 / (1 + exp(1 + age - 2*gender - 0.5*mud - 0.5*nud)); mdd = sapply(mdd, function(prob) rbinom(1, 1, prob))
  
  
  if(extended) {
    
    # jeweils 10 items (0,1)
    # cutoff ist jeweils >=6
    
    mddvals = lapply(mdd,function(x) {
      
      if (x==1) {
        count= sample(6:10,1)
      }
      if (x==0) {
        count= sample(0:5,1)
      }
      ind = sample(1:10,count)
      vec = rep(0,10)
      vec[ind] =1
      return(vec)
    })
    
    mddx = do.call(rbind,mddvals)
    
    
    nudvals = lapply(nud,function(x) {
      
      if (x==1) {
        count= sample(6:10,1)
      }
      if (x==0) {
        count= sample(0:5,1)
      }
      ind = sample(1:10,count)
      vec = rep(0,10)
      vec[ind] =1
      return(vec)
    })
    
    nudx = do.call(rbind,nudvals)
    
    
    mudvals = lapply(mud,function(x) {
      
      if (x==1) {
        count= sample(6:10,1)
      }
      if (x==0) {
        count= sample(0:5,1)
      }
      ind = sample(1:10,count)
      vec = rep(0,10)
      vec[ind] =1
      return(vec)
    })
    
    mudx = do.call(rbind,mudvals)
  
    df=cbind(gender,age,mddx,nudx,mudx)
      
    colnames(df) = c("gender","age",paste0(c("mdd"),1:10),paste0(c("nud"),1:10),paste0(c("mud"),1:10))
    
  }
  
  
  
  if(!extended) {
  
  df = data.frame(gender, age, mdd, nud, mud)
  }
  
  # Simulate alcohol use disorder from highly nonlinear model.
  z = -1 - (age + age**2 + age**3 + age**4 + age**5) +
    2*gender + 0.5*mdd + 0.5*nud + 0.5*mud -
    2*(age + age**2 + age**3 + age**4 + age**5)*gender
  aud = 1 / (1 + exp(-z))
  aud = sapply(aud, function(prob) rbinom(1, 1, prob))
  
  
  # dat = cbind(df,aud)
  dat = list(y =aud, x = df)
  # return(list(y =aud, x = df))
  
  return(dat)
  
}

#' Title
#'
#' @param sample_size 
#'
#' @return
#' @export
#'
#' @examples
data_conspiracy = function(n,preprocessing=TRUE) {
  
  realistic = FALSE
  
  if (realistic) {
    #realistic, discrete version
    
    agemean= 30
    agesd = 10
    cutoff = 18
    age = runif(n,pnorm(cutoff,agemean,agesd),1)
    age = round(qnorm(age,agemean,agesd)) # 18-open end
    
    edu = 1+rbinom(n, 4,sigmoid(scale(age)/3)) # 1-5 (Hauptschule -> Uni)
    
    polori = 1+rbinom(n, 8,sigmoid(scale(age)/10)) # 1-9 (links -> rechts)
    
    cm = 5+rbinom(n, 20,sigmoid(scale(age)/10+scale((scale(polori)+.4)^2)+scale(polori)/6-scale(edu)/9)) #5 items from likert scale
    
  }
  
  if (!realistic) {
    #real-valued version
    
    agemean= 30
    agesd = 10
    cutoff = 18
    age = runif(n,pnorm(cutoff,agemean,agesd),1) 
    age = qnorm(age,agemean,agesd) %>% scale() 
    
    edu = rnorm(n,mean= scale(age)/3) %>% scale()
    polori  = rnorm(n,mean=sigmoid(scale(age)/10)) %>% scale()
    
    # cm = rnorm(n,mean =sigmoid(scale(age)/10+scale((scale(polori)+.4)^2)+scale(polori)/6-scale(edu)/9) )
    cm = 5+rbinom(n, 20,sigmoid(scale(age)/10+scale((scale(polori)+.4)^2)+scale(polori)/6-scale(edu)/9+rnorm(n,sd=.5))) #5 items from likert scale
    
  }
  
  
  dat = data.frame(age,edu,polori,cm)
  
  if (preprocessing) {
    dat$polori = scale(dat$polori) %>% as.numeric()
    
    dat$polori2 = dat$polori^2
  }
  
  return(dat)
}


#' Title
#'
#' @param type 
#'
#' @return
#' @export
#'
#' @examples
data_fashion = function(type="train") {

dataset <- dataset_fashion_mnist()

if (type=="train") dat = dataset$train
if (type=="test") dat = dataset$test

dat$x <- dat$x / 255

return(dat)
# if (type=="test") {
#   dat_train = dat$train
#   dat_train$x <- dat_train$x / 255
# }
# 
# 
# 
# dat_train = dat$train
# dat_test = dat$test
# 
# class_names = c('T-shirt/top',
#                 'Trouser',
#                 'Pullover',
#                 'Dress',
#                 'Coat', 
#                 'Sandal',
#                 'Shirt',
#                 'Sneaker',
#                 'Bag',
#                 'Ankle boot')
# 
# dat_train$x <- dat_train$x / 255
# dat_test$x <- dat_test$x / 255

}

# learner -----------------------------------------------------------------

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
train_nn = function(mod,x,y,loss,epochs = 20,learning_rate=.001,optimizer="adam",batch_size=nrow(x),metrics=NULL,silent=FALSE,early_stopping=FALSE,validation_split = .2) {
  
  if(is.data.frame(x)) x = as.matrix(x)
  if(is.data.frame(y)) y = as.matrix(y)
  
  if(tolower(optimizer)=="adam") optim = optimizer_adam(learning_rate)
  if(tolower(optimizer)=="rmsprop") optim = optimizer_rmsprop(learning_rate)
  if(tolower(optimizer)=="sgd") optim = optimizer_sgd(learning_rate)
  
  if (tolower(loss)=="ce") loss = "binary_crossentropy"
  
  if (tolower(loss)=="cce") {
    # sparse cce
    if(length(dim(y)) == 1) loss = "sparse_categorical_crossentropy"
    # normal cce
    if(length(dim(y)) > 1) loss = "categorical_crossentropy"
  }
  
  if(validation_split==0) monitor = "loss"
  if(validation_split>0) monitor = "val_loss"
  
  if (isTRUE(early_stopping))early_stopping=5
  if (isFALSE(early_stopping)||early_stopping==0){
    callbacks_list = NULL
    validation_split=0
  }
  
  if (early_stopping) {
    callbacks_list = list(
      callback_early_stopping(
        monitor = monitor,
        patience = early_stopping
      ),
      callback_model_checkpoint(
        filepath = "mymodel.h5",
        monitor = monitor, save_best_only = TRUE
      )
    )
  }
  
  
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
    callbacks = callbacks_list,
    validation_split = validation_split,
    verbose=verbosity
  )
  
  if(verbosity==0)   {
    
    final_loss = history$metrics$loss[length(history$metrics$loss)]
    print(paste0("Final Loss (",loss,"): ",round(final_loss,7)))
  }
  
  return(mod)
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
    rownames(temp) = paste0("in",1:nrow(temp))
    if(i==ind_weights[1]) rownames(temp) = paste0("pred",1:nrow(temp))
    colnames(temp) = paste0("neuron",1:ncol(temp))
    a[[i]] = temp
  }
  
  return(a)
  
}



#' Title
#'
#' @param x 
#' @param y 
#'
#' @return
#' @export
#'
#' @examples
train_featureless = function(x,y) {
  
  y_onehot <- to_categorical(y)
  re = colMeans(y_onehot)
  
  class(re) = "featureless"
  
  return(re)  
}


#' Title
#'
#' @param mod 
#' @param x 
#'
#' @return
#' @export
#'
#' @examples
predict.featureless = function(mod,x) {
  
  mod %>% rep(.,times=nrow(x)) %>% matrix(.,nrow=nrow(x))
  
}

