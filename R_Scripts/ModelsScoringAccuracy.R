library(dplyr)
library(tidyverse)
library(dplyr)
library(splines)
library(survminer)
library(survival)
library(lubridate)
library(shinythemes)
library(shiny)
library(ggplot2)
library(lazyeval)
library(tibble)
library(qdapRegex)
# have to determine 15 to 19
setwd("")
combination_number<-1
Scorer_Name<-"Scorer1"
Image_Set<-"practice"
Data<-Eval_Data_All
  dm<-dim(Data)
RowNumber=dm[1]

PersonToCompare<-1
Data2<-data.frame(Data$Model, Data$Filename, Data$Folder,Data$Status)
Data2$Model<-as.character(Data2$Model)
Data2$Data.Status<-as.character(Data2$Data.Status)
Data2$Data.Folder<-as.character(Data2$Data.Folder)
Data2$Data.Filename<-as.character(Data2$Data.Filename)




a<-Data$Model
Model<-unique(a)

c<-length(Model)
d<-Data2$Data.Folder
Scorer<-unique(d)
f<-length(Scorer)



#getting name of image     rm_between(Data2[1,2],"/",".",extract=TRUE)



Data2[,5]<-as.character(Scorer[PersonToCompare])


Data2[,6]<-as.character(substr(Data2$Data.Filename,1,1))

Data2[,7]<-as.character(ifelse(Data2[,5]!= Data2$Data.Folder,NA,ifelse(Data2[,6]=="A",0,1)))

for (i in 1:c){
  Data2[,i+7]<-as.character(Model[i])
}
for (i in 1:c){
  Data2[,i+7+c]<-as.character(ifelse(Data2[,i+7]==Data2$Data.Model & Data2$Data.Status=="FALSE",1,ifelse(Data2[,i+7]==Data2$Data.Model & Data2$Data.Status=="TRUE",0,NA)))
}

for (i in 1:c){
  Data2[,i+7+2*c]<-as.numeric(ifelse(Data2[,i+7+c]==Data2[,7],1,0))
}
ImageID_Scorer<-ifelse(Data2[,5]==Data2$Data.Folder,1,0)
Total_Images_Scored_Per_Model<-(sum(ImageID_Scorer))/c

for (i in 1:c){
  Data2[1,i+7+3*c]<-sum(Data2[,i+7+2*c],na.rm = TRUE)
}

for (i in 1:c){
  Data2[1,i+7+4*c]<-Data2[1,i+7+3*c]/Total_Images_Scored_Per_Model*100
}


#Alive by person as Alive by computer

#Alive by manual

Data2[,8+5*c]<-ifelse(Data2[,7]==0,1,NA)

Data2[1,9+5*c]<-sum(Data2[,8+5*c],na.rm = TRUE)

Data2[1,10+5*c]<-Data2[1,9+5*c]/c

Manual_Alive<-as.numeric(Data2[1,10+5*c])
#agreement of those marked alive manually and marked alive by Neural Network
for (i in 1:c){
  Data2[,i+10+5*c]<-ifelse(Data2[,7]==0,Data2[,i+7+2*c],NA)
}

for (i in 1:c){
  Data2[1,i+10+6*c]<-sum(Data2[,i+10+5*c],na.rm = TRUE)
}

#those marked alive manually and dead by Neural Network

for (i in 1:c){
  Data2[1,i+10+7*c]<-Data2[1,10+5*c]-Data2[1,i+10+6*c]
}

#those marked dead manually 

Data2[,11+8*c]<-ifelse(Data2[,7]==1,1,NA)

Data2[1,12+8*c]<-sum(Data2[,11+8*c],na.rm = TRUE)

Data2[1,13+8*c]<-Data2[1,12+8*c]/c

Manual_Dead<-as.numeric(Data2[1,13+8*c])

#those marked dead manually and dead by neural network

for (i in 1:c){
  Data2[,i+13+8*c]<-ifelse(Data2[,7]==1,Data2[,i+7+2*c],NA)
}

for (i in 1:c){
  Data2[1,i+13+9*c]<-sum(Data2[,i+13+8*c],na.rm = TRUE)
}

#those marked dead manually and alive by neural network

for (i in 1:c){
  Data2[1,i+13+10*c]<-Data2[1,13+8*c]-Data2[1,i+13+9*c]
}

#percentages of alive manual marked alive by neural network

for (i in 1:c){
  Data2[1,i+13+11*c]<-Data2[1,i+10+6*c]/Data2[1,10+5*c]*100
}

#percentages of alive manual marked dead by neural network

for (i in 1:c){
  Data2[1,i+13+12*c]<-Data2[1,i+10+7*c]/Data2[1,10+5*c]*100
}

#percentages of dead manual marked dead by neural network

for (i in 1:c){
  Data2[1,i+13+13*c]<-Data2[1,i+13+9*c]/Data2[1,13+8*c]*100
}

#percentages of dead manual marked dead by neural network

for (i in 1:c){
  Data2[1,i+13+14*c]<-Data2[1,i+13+10*c]/Data2[1,13+8*c]*100
}


#write.csv(Data2, "Data2_trialb.csv")
dmData2<-dim(Data2)
rowData2<-dmData2[1]
colData2<-dmData2[2]

accuracy<-matrix(0,ncol=c, nrow = 11)
Accuracy<-as.data.frame(accuracy)
for (i in 1:c){
  colnames(Accuracy)[i]<-Model[i]
}

row.names(Accuracy)[1]<-paste0("Manual(",Scorer_Name,") Scored Alive Scored Alive by Neural Network")
row.names(Accuracy)[2]<-"Manual Scored Alive Scored Dead by Neural Network"
row.names(Accuracy)[3]<-"Manual Alive Scored"
row.names(Accuracy)[4]<-"Manual Scored Dead Scored Dead by Neural Network"
row.names(Accuracy)[5]<-"Manual Scored Dead Scored Alive by Neural Network"
row.names(Accuracy)[6]<-"Manual Dead Scored"
row.names(Accuracy)[7]<-"Percent Accuracy of All Cells"
row.names(Accuracy)[8]<-"Percent Manual Scored Alive Scored Alive by Neural Network"
row.names(Accuracy)[9]<-"Percent Manual Scored Alive Scored Dead by Neural Network"
row.names(Accuracy)[10]<-"Percent Manual Scored Dead Scored Dead by Neural Network"
row.names(Accuracy)[11]<-"Percent Manual Scored Dead Scored Alive by Neural Network"

for (i in 1:c){
  Accuracy[1,i]<-paste(Data2[1,i+10+6*c])
}

for (i in 1:c){
  Accuracy[2,i]<-paste(Data2[1,i+10+7*c])
}

for (i in 1:c){
  Accuracy[3,i]<-paste(Data2[1,10+5*c])
}


for (i in 1:c){
  Accuracy[4,i]<-paste(Data2[1,i+13+9*c])
}

for (i in 1:c){
  Accuracy[5,i]<-paste(Data2[1,i+13+10*c])
}

for (i in 1:c){
  Accuracy[6,i]<-paste(Data2[1,13+8*c])
}

for (i in 1:c){
  Accuracy[7,i]<-paste(Data2[1,i+7+4*c])
}

for (i in 1:c){
  Accuracy[8,i]<-paste(Data2[1,i+13+11*c])
}


for (i in 1:c){
  Accuracy[9,i]<-paste(Data2[1,i+13+12*c])
}


for (i in 1:c){
  Accuracy[10,i]<-paste(Data2[1,i+13+13*c])
}


for (i in 1:c){
  Accuracy[11,i]<-paste(Data2[1,i+13+14*c])
}




#write.csv(Accuracy,paste("Accuracy_for_Individual_Models_Compared_To_",Scorer_Name,"_",Image_Set,".csv"))

Data2matrix<-matrix(0,ncol=colData2, nrow = rowData2)
Data3<-as.data.frame(Data2matrix)
rrr<-sub("^","Cell_Status",Model)


for (i in 1:c){
  colnames(Data3)[i+7+c]<-rrr[i]
  
  
}

for (i in 1:c){
  Data3[,i+7+c]<-paste(Data2[,i+7+c])
}

colnames(Data3)[1]<-"Model_Used"
colnames(Data3)[2]<-"File_Evaluated"
colnames(Data3)[3]<-"Folder"
colnames(Data3)[4]<-"Status_Provided_by_Model"
colnames(Data3)[5]<-"Person_Comparing_To"
colnames(Data3)[6]<-"Status_Provided_by_Person"
colnames(Data3)[7]<-"Status_by_Person_Numerical"

for (i in 1:7){
  Data3[,i]<-paste(Data2[,i])
}

for (i in 1:c){
  Data3[,i+7]<-Data2[,i+7]
}




sss<-sub("^","Agreement(1)_with_Person_",Model)





for (i in 1:c){
  colnames(Data3)[i+7+2*c]<-sss[i]
}

for (i in 1:c){
  Data3[,i+7+2*c]<-Data2[,i+7+2*c]
}


tt<-sub("^","Sum_ofAgreement_withPerson",Model)
uu<-sub("^","Percent_Agreement_withPerson",Model)

for (i in 1:c){
  colnames(Data3)[i+7+3*c]<-tt[i]
}

for (i in 1:c){
  colnames(Data3)[i+7+4*c]<-uu[i]
}



for (i in 1:c){
  Data3[1,i+7+3*c]<-sum(Data3[,i+7+2*c],na.rm = TRUE)
}

for (i in 1:c){
  Data3[1,i+7+4*c]<-Data3[1,i+7+3*c]/Total_Images_Scored_Per_Model
}







#Individual Scoring

m<-matrix(0,ncol=c, nrow =1 )
Individual_Scoring<-as.data.frame(m)
for (i in 1:c){
  colnames(Individual_Scoring)[i]<-Model[i]
  
  
}

for (i in 1:c){
  Individual_Scoring[1,i]<-paste(Data2[1,i+7+4*c])
}
gg<-Scorer_Name
ff<-paste("Accuracy_Compared to ",gg, sep="")

rownames(Individual_Scoring)[1]<-ff
#write.csv(Individual_Scoring, paste0("Scoring_Accuracy_of_Individual_Models_Compared_To",gg,Image_Set,".csv"))

# Combination Scoring for both positive and negative accuracy





mm<-matrix(0,ncol=c, nrow =dm )
Agreement_with_Manual<-as.data.frame(mm)

for (i in 1:c){
  Agreement_with_Manual[,i]<-paste(Data2[,i+7+2*c])
}


for (i in 1:c){
  colnames(Agreement_with_Manual)[i]<-Model[i]
  
  
}

Agreement_with_Manual$Data.Filename<-Data2$Data.Filename
Agreement_with_Manual$Filename<-rm_between(Agreement_with_Manual$Data.Filename,"/",".",extract=TRUE)
www<-RowNumber/(c*f)

mmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted<-as.data.frame(mmm)

for (i in 1:c){
  colnames(Agreement_with_Manual_Sorted)[i]<-sss[i]
}

for (i in 1:c){
  ifelse(i==1,Agreement_with_Manual_Sorted[1:www,i]<-Agreement_with_Manual[1:www,i],Agreement_with_Manual_Sorted[1:www,i]<-Agreement_with_Manual[(((i-1)*www)+1):(www*i),i])
}


#write.csv(Agreement_with_Manual_Sorted, "Agreement_with_Manual_Sorted.csv")


mmmmmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted2<-data.frame(mmmmmm)

for (i in 1:c){
  Agreement_with_Manual_Sorted2[,i]<-as.numeric(Agreement_with_Manual_Sorted[,i])
}


k<- combn(Agreement_with_Manual_Sorted2, m = combination_number, FUN = rowSums, simplify = TRUE)
j<-combn(Model,m = combination_number, FUN = NULL, simplify = TRUE )
jj<-as.data.frame(j)
xy<-dim(j)
xx<-xy[2]
zz<-xy[1]
zzz<-zz+1
possiblecombinations<-xx

Sum_of_Agreements_And_Disagreements<-as.data.frame(k)
#write.csv(Sum_of_Agreements_And_Disagreements,"Sum_of_Agreements_And_Disagreements.csv")
Average_of_Agreements_And_Disagreements<-as.data.frame(k/combination_number)
#write.csv(Average_of_Agreements_And_Disagreements,"Average_of_Agreements_and_Disagreement.csv")
Combinations_Agree_orDisagree<-as.data.frame(ifelse(Average_of_Agreements_And_Disagreements>0.5,1,0))
#write.csv(Combinations_Agree_orDisagree, "Combinations_Agree_orDisagree.csv")

mmmmmmm<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Agree_orDisagree_Sum<-data.frame(mmmmmmm)


for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Sum[1,i]<-sum(Combinations_Agree_orDisagree[,i])
}

#write.csv(Combinations_Agree_orDisagree_Sum, "Combinations_Agree_or_Disagree_Sum")



Combinations_Agree_orDisagree_Mean_Accuracy<-data.frame()

for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy[2,i]<-Combinations_Agree_orDisagree_Sum[,i]/Total_Images_Scored_Per_Model*100
  
}

combomodel<-matrix(0,ncol=possiblecombinations, nrow =1 )
Names_Model_Combo<-data.frame(combomodel)
for (i in 1:possiblecombinations){
  combomodel[1,i]<-paste(j[1:combination_number,i],collapse = "")
  
}
for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy[1,i]<-combomodel[,i]
  
}


#write.csv(Combinations_Agree_orDisagree_Mean_Accuracy,paste0("Scoring_Accuracy_Overall",combination_number,"_model_combinations_compared_to",gg,Image_Set,".csv"))

#combination scoring for alive cells




alive<-matrix(0,ncol=possiblecombinations, nrow =dm )
Agreement_with_Manual_alive<-as.data.frame(alive)

for (i in 1:c){
  Agreement_with_Manual_alive[,i]<-paste( Data2[,i+10+5*c])
}

for (i in 1:c){
  colnames(Agreement_with_Manual_alive)[i]<-Model[i]
  
  
}

Agreement_with_Manual_alive$Data.Filename<-Data2$Data.Filename
Agreement_with_Manual_alive$Filename<-rm_between(Agreement_with_Manual_alive$Data.Filename,"/",".",extract=TRUE)
www<-RowNumber/(c*f)

alivesorted<-matrix(0,ncol=c, nrow = www )
Agreement_with_Manual_alive_Sorted<-as.data.frame(alivesorted)

for (i in 1:c){
  colnames(Agreement_with_Manual_alive_Sorted)[i]<-sss[i]
}

for (i in 1:c){
  ifelse(i==1,Agreement_with_Manual_alive_Sorted[1:www,i]<-Agreement_with_Manual_alive[1:www,i],Agreement_with_Manual_alive_Sorted[1:www,i]<-Agreement_with_Manual_alive[(((i-1)*www)+1):(www*i),i])
}




#write.csv(Agreement_with_Manual_alive_Sorted, "Agreement_with_Manual_Alive_Sorted.csv")

mmmmmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted_aliveB<-data.frame(mmmmmm)

for (i in 1:c){
  Agreement_with_Manual_Sorted_aliveB[,i]<-ifelse(Agreement_with_Manual_alive_Sorted[,i]=="NA","",Agreement_with_Manual_alive_Sorted[,i])
}

for (i in 1:c){
  colnames(Agreement_with_Manual_Sorted_aliveB)[i]<-sss[i]
}



#write.csv(Agreement_with_Manual_Sorted_aliveB, "Agreement_with_Manual_Alive_SortedB.csv")

mmmmmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted_aliveC<-data.frame(mmmmmm)

for (i in 1:c){
  Agreement_with_Manual_Sorted_aliveC[,i]<-as.numeric(Agreement_with_Manual_Sorted_aliveB[,i])
}


k<- combn(Agreement_with_Manual_Sorted_aliveC, m = combination_number, FUN = rowSums, simplify = TRUE)
j<-combn(Model,m = combination_number, FUN = NULL, simplify = TRUE )
jj<-as.data.frame(j)
xy<-dim(j)
xx<-xy[2]
zz<-xy[1]
zzz<-zz+1
possiblecombinations<-xx

Sum_of_Agreements_And_Disagreements_Alive<-as.data.frame(k)
#write.csv(Sum_of_Agreements_And_Disagreements_Alive,"Sum_of_Agreements_And_Disagreements_Alive.csv")
Average_of_Agreements_And_Disagreements_Alive<-as.data.frame(k/combination_number)
#write.csv(Average_of_Agreements_And_Disagreements_Alive,"Average_of_Agreements_and_Disagreement_Alive.csv")
AandDalive<-matrix(0,ncol=possiblecombinations, nrow =www)
Average_of_Agreements_And_DisagreementsB_alive<-data.frame(AandDalive)

for (i in 1:possiblecombinations){
  Average_of_Agreements_And_DisagreementsB_alive[,i]<-Average_of_Agreements_And_Disagreements_Alive[,i]
}
Average_of_Agreements_And_DisagreementsB_alive[is.na(Average_of_Agreements_And_DisagreementsB_alive)]<-""

#write.csv(Average_of_Agreements_And_DisagreementsB_alive,"Average_of_Agreements_and_Disagreement_AliveB.csv")
Combinations_Agree_orDisagree_Alive<-as.data.frame(ifelse(Average_of_Agreements_And_DisagreementsB_alive>0.5,1,0))
#write.csv(Combinations_Agree_orDisagree_Alive, "Combinations_Agree_orDisagree_Alive.csv")

mmmmmmm<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Agree_orDisagree_Sum_Alive<-data.frame(mmmmmmm)


for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Sum_Alive[1,i]<-sum(Combinations_Agree_orDisagree_Alive[,i])
}

#write.csv(Combinations_Agree_orDisagree_Sum_Alive, "Combinations_Agree_or_Disagree_Sum_Alive.csv")



Combinations_Agree_orDisagree_Mean_Accuracy_Alive<-data.frame()

for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy_Alive[2,i]<-Combinations_Agree_orDisagree_Sum_Alive[,i]/Manual_Alive
  
}

combomodel<-matrix(0,ncol=possiblecombinations, nrow =1 )
Names_Model_Combo<-data.frame(combomodel)
for (i in 1:possiblecombinations){
  combomodel[1,i]<-paste(j[1:combination_number,i],collapse = "")
  
}
for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy_Alive[1,i]<-combomodel[,i]
  
}


#write.csv(Combinations_Agree_orDisagree_Mean_Accuracy_Alive,paste0("Scoring_Accuracy_Alive",combination_number,"_model_combinations_compared_to",gg,Image_Set,".csv"))

combomodel2<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Disagree_ManualAlive<-data.frame(combomodel2)

for (i in 1:possiblecombinations){
  Combinations_Disagree_ManualAlive[,i]<-Manual_Alive- Combinations_Agree_orDisagree_Sum_Alive[,i]
}

combomodel3<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Disagree_ManualAlivePercent<-data.frame(combomodel3)

for (i in 1:possiblecombinations){
  Combinations_Disagree_ManualAlivePercent[,i]<-Combinations_Disagree_ManualAlive[,i]/Manual_Alive*100
}

#scoring accuracy combinations for dead


dead<-matrix(0,ncol=possiblecombinations, nrow =dm )
Agreement_with_Manual_dead<-as.data.frame(alive)

for (i in 1:c){
  Agreement_with_Manual_dead[,i]<-paste(Data2[,i+13+8*c])
}

for (i in 1:c){
  colnames(Agreement_with_Manual_dead)[i]<-Model[i]
  
  
}

Agreement_with_Manual_dead$Data.Filename<-Data2$Data.Filename
Agreement_with_Manual_dead$Filename<-rm_between(Agreement_with_Manual_dead$Data.Filename,"/",".",extract=TRUE)
www<-RowNumber/(c*f)

deadsorted<-matrix(0,ncol=c, nrow = www )
Agreement_with_Manual_dead_sorted<-as.data.frame(deadsorted)

for (i in 1:c){
  colnames(Agreement_with_Manual_dead_sorted)[i]<-sss[i]
}

for (i in 1:c){
  ifelse(i==1,Agreement_with_Manual_dead_sorted[1:www,i]<-Agreement_with_Manual_dead[1:www,i],Agreement_with_Manual_dead_sorted[1:www,i]<-Agreement_with_Manual_dead[(((i-1)*www)+1):(www*i),i])
}




#write.csv(Agreement_with_Manual_dead_sorted, "Agreement_with_Manual_Dead_Sorted.csv")

mmmmmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted_deadB<-data.frame(mmmmmm)

for (i in 1:c){
  Agreement_with_Manual_Sorted_deadB[,i]<-ifelse(Agreement_with_Manual_dead_sorted[,i]=="NA","",Agreement_with_Manual_dead_sorted[,i])
}

for (i in 1:c){
  colnames(Agreement_with_Manual_Sorted_deadB)[i]<-sss[i]
}



#write.csv(Agreement_with_Manual_Sorted_deadB, "Agreement_with_Manual_dead_SortedB.csv")

mmmmmm<-matrix(0,ncol=c, nrow =www )
Agreement_with_Manual_Sorted_deadC<-data.frame(mmmmmm)

for (i in 1:c){
  Agreement_with_Manual_Sorted_deadC[,i]<-as.numeric(Agreement_with_Manual_Sorted_deadB[,i])
}


k<- combn(Agreement_with_Manual_Sorted_deadC, m = combination_number, FUN = rowSums, simplify = TRUE)
j<-combn(Model,m = combination_number, FUN = NULL, simplify = TRUE )
jj<-as.data.frame(j)
xy<-dim(j)
xx<-xy[2]
zz<-xy[1]
zzz<-zz+1
possiblecombinations<-xx

Sum_of_Agreements_And_Disagreements_Dead<-as.data.frame(k)
#write.csv(Sum_of_Agreements_And_Disagreements_Dead,"Sum_of_Agreements_And_Disagreements_Dead.csv")
Average_of_Agreements_And_Disagreements_Dead<-as.data.frame(k/combination_number)
#write.csv(Average_of_Agreements_And_Disagreements_Dead,"Average_of_Agreements_and_Disagreement_Dead.csv")
AandD_dead<-matrix(0,ncol=possiblecombinations, nrow =www)
Average_of_Agreements_And_DisagreementsB_dead<-data.frame(AandD_dead)

for (i in 1:possiblecombinations){
  Average_of_Agreements_And_DisagreementsB_dead[,i]<-Average_of_Agreements_And_Disagreements_Dead[,i]
}

Average_of_Agreements_And_DisagreementsB_dead[is.na(Average_of_Agreements_And_DisagreementsB_dead)]<-""

#write.csv(Average_of_Agreements_And_DisagreementsB_dead,"Average_of_Agreements_and_Disagreement_Dead_B.csv")
Combinations_Agree_orDisagree_Dead<-as.data.frame(ifelse(Average_of_Agreements_And_DisagreementsB_dead>0.5,1,0))
#write.csv(Combinations_Agree_orDisagree_Dead, "Combinations_Agree_orDisagree_Dead.csv")

mmmmmmm<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Agree_orDisagree_Sum_Dead<-data.frame(mmmmmmm)


for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Sum_Dead[1,i]<-sum(Combinations_Agree_orDisagree_Dead[,i])
}

#write.csv(Combinations_Agree_orDisagree_Sum_Dead, "Combinations_Agree_or_Disagree_Sum_Dead.csv")



Combinations_Agree_orDisagree_Mean_Accuracy_Dead<-data.frame()

for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy_Dead[2,i]<-Combinations_Agree_orDisagree_Sum_Dead[,i]/Manual_Dead
  
}

combomodel<-matrix(0,ncol=possiblecombinations, nrow =1 )
Names_Model_Combo<-data.frame(combomodel)
for (i in 1:possiblecombinations){
  combomodel[1,i]<-paste(j[1:combination_number,i],collapse = "")
  
}
for (i in 1:possiblecombinations){
  Combinations_Agree_orDisagree_Mean_Accuracy_Dead[1,i]<-combomodel[,i]
  
}



#write.csv(Combinations_Agree_orDisagree_Mean_Accuracy_Dead,paste0("Scoring_Accuracy_Dead",combination_number,"_model_combinations_compared_to",gg,Image_Set,".csv"))


combomodel9<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Disagree_ManualDead<-data.frame(combomodel9)

for (i in 1:possiblecombinations){
  Combinations_Disagree_ManualDead[1,i]<-Manual_Dead-Combinations_Agree_orDisagree_Sum_Dead[,i]
  
}

combomodel10<-matrix(0,ncol=possiblecombinations, nrow =1 )
Combinations_Disagree_ManualDead_Percent<-data.frame(combomodel10)

for (i in 1:possiblecombinations){
  Combinations_Disagree_ManualDead_Percent[1,i]<-Combinations_Disagree_ManualDead[,i]/Manual_Dead*100
  
}

combomodel11<-matrix(0,ncol=possiblecombinations, nrow =1 )
Precision<-data.frame(combomodel11)

for (i in 1:possiblecombinations){
  Precision[1,i]<-Manual_Alive/(Manual_Alive+Combinations_Disagree_ManualDead[,i])
  
}


combomodel12<-matrix(0,ncol=possiblecombinations, nrow =1 )
Recall_<-data.frame(combomodel12)

for (i in 1:possiblecombinations){
  Recall_[1,i]<-Manual_Alive/(Manual_Alive+Manual_Alive-Combinations_Agree_orDisagree_Sum_Alive[1,i])
  
}

combomodel13<-matrix(0,ncol=possiblecombinations, nrow =1 )
F1_score<-data.frame(combomodel13)

for (i in 1:possiblecombinations){
  F1_score[1,i]<-2*Precision[1,i]*Recall_[1,i]/(Precision[1,i]+Recall_[1,i])
  
}




#output for combinations of correct and incorrect alive and dead

accuracycombo<-matrix(0,ncol=possiblecombinations, nrow = 15)
AccuracyCombination<-as.data.frame(accuracycombo)
for (i in 1:c){
  colnames(Accuracy)[i]<-Model[i]
}

row.names(AccuracyCombination)[1]<-"Neural Networks Combined"
row.names(AccuracyCombination)[2]<-paste0("Manual(",Scorer_Name,") Scored Alive Scored Alive by Neural Networks")
row.names(AccuracyCombination)[3]<-"Manual Scored Alive Scored Dead by Neural Networks"
row.names(AccuracyCombination)[4]<-"Manual Alive Scored"
row.names(AccuracyCombination)[5]<-"Manual Scored Dead Scored Dead by Neural Networks"
row.names(AccuracyCombination)[6]<-"Manual Scored Dead Scored Alive by Neural Networks"
row.names(AccuracyCombination)[7]<-"Manual Dead Scored"
row.names(AccuracyCombination)[8]<-"Percent Accuracy of All Cells"
row.names(AccuracyCombination)[9]<-"Percent Manual Scored Alive Scored Alive by Neural Networks"
row.names(AccuracyCombination)[10]<-"Percent Manual Scored Alive Scored Dead by Neural Networks"
row.names(AccuracyCombination)[11]<-"Percent Manual Scored Dead Scored Dead by Neural Networks"
row.names(AccuracyCombination)[12]<-"Percent Manual Scored Dead Scored Alive by Neural Networks"
row.names(AccuracyCombination)[13]<-"Precision"
row.names(AccuracyCombination)[14]<-"Recall"
row.names(AccuracyCombination)[15]<-"F1_score"

for (i in 1:possiblecombinations){
  AccuracyCombination[1,i]<-paste(j[1:combination_number,i],collapse = "")
}


for (i in 1:possiblecombinations){
  AccuracyCombination[2,i]<-paste(Combinations_Agree_orDisagree_Sum_Alive[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[3,i]<-as.numeric(Manual_Alive-Combinations_Agree_orDisagree_Sum_Alive[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[4,i]<-paste(Manual_Alive)
}


for (i in 1:possiblecombinations){
  AccuracyCombination[5,i]<-paste(Combinations_Agree_orDisagree_Sum_Dead[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[6,i]<-paste(Manual_Dead-Combinations_Agree_orDisagree_Sum_Dead[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[7,i]<-paste(Manual_Dead)
}

for (i in 1:possiblecombinations){
  AccuracyCombination[8,i]<-paste(Combinations_Agree_orDisagree_Mean_Accuracy[2,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[9,i]<-paste(Combinations_Agree_orDisagree_Sum_Alive[1,i]/Manual_Alive*100)
}


for (i in 1:possiblecombinations){
  AccuracyCombination[10,i]<-paste(Combinations_Disagree_ManualAlivePercent[1,i])
}


for (i in 1:possiblecombinations){
  AccuracyCombination[11,i]<-paste(Combinations_Agree_orDisagree_Sum_Dead[1,i]/Manual_Dead*100)
}


for (i in 1:possiblecombinations){
  AccuracyCombination[12,i]<-paste(Combinations_Disagree_ManualDead_Percent[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[13,i]<-paste(Precision[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[14,i]<-paste(Recall_[1,i])
}

for (i in 1:possiblecombinations){
  AccuracyCombination[15,i]<-paste(F1_score[1,i])
}





write.csv(AccuracyCombination,paste("Accuracy_for_Combination_of_",combination_number,"_Models_Compared_To_",Scorer_Name,'_',Image_Set,".csv"))
