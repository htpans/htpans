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




# DIRECTIONS: Only input needed is lines 19 to 26. 

setwd("")
Data_Input_Sheet<-Excel_sample_sheet
Experiment<-"Sample"
Reference_Group<-"Group1"
x_axisLabel<-"Hours"
x_axisScale<-24
time_interval<-8
groupsexamine<-c("Group1","Group2","Group3","Group4")
Scoring<- subset(Data_Input_Sheet, select=c("Image", "Cell", "Time", "Status","ImagingPlate","transfection","DrugPlate","Drug","Class"))
Scoring$Alive<-ifelse(Scoring$Status=="TRUE",1,2)
Scoring$Well_ROI<-paste(Scoring$Image,Scoring$Cell, sep="_")
Scoring$Well_ROI_ImagingPlate<-paste(Scoring$Well_ROI,Scoring$ImagingPlate,sep="_")
Scoring$TimeAdjusted <- Scoring$Time -1 
Scoring$TimeAdjusted<-Scoring$TimeAdjusted*time_interval
Scoring$NplusOneTimepoint<- append(Scoring$TimeAdjusted[-1], 0, length(Scoring$TimeAdjusted-1))
Scoring$IsFirstImage <-ifelse(Scoring$TimeAdjusted==0,"FIRST","")
Scoring$NplusOneScore<-append(Scoring$Alive[-1], 0, length(Scoring$Alive-1))
d<- append(Scoring$Alive, 0, after=0)
Scoring$NminusOneScore<-d[1:length(d)-1]
e<- append(Scoring$NminusOneScore,0, after=0)
Scoring$NminusTwoScore<-e[1:length(e)-1]
Scoring$Sum3TimepointsforMiddle <- rowSums(cbind(Scoring$Alive,Scoring$NminusOneScore,Scoring$NplusOneScore))
Scoring$Sum2TimepointsforFirst<- rowSums(cbind(Scoring$Alive,Scoring$NplusOneScore))
Scoring$Sum3TimepointsforLast<-rowSums(cbind(Scoring$Alive,Scoring$NminusOneScore,Scoring$NminusTwoScore))
Scoring$CellPositionID_Last<-ifelse(Scoring$NplusOneTimepoint==0,"LAST","")
Scoring$CellPositionID_FirstandLast <- paste(Scoring$CellPositionID_Last,Scoring$IsFirstImage)
Scoring$CellPostion<-ifelse(Scoring$IsFirstImage=="FIRST"|Scoring$CellPositionID_Last=="LAST",Scoring$CellPositionID_FirstandLast,"MIDDLE")
Scoring$MiddleStatus<-ifelse(Scoring$CellPostion=="MIDDLE" & Scoring$Sum3TimepointsforMiddle>=5, 1,0)
Scoring$FirstStatus<-ifelse(Scoring$CellPostion==" FIRST" & Scoring$Sum2TimepointsforFirst>=4, 1,0)
Scoring$LastStatus<-ifelse(Scoring$CellPositionID_Last=="LAST" & Scoring$Alive ==2, 1,0)
Scoring$ScoringALL<-rowSums(cbind(Scoring$FirstStatus,Scoring$MiddleStatus,Scoring$LastStatus))
Scoring$ImageTimeDeadFirst<-ifelse(Scoring$CellPostion==" FIRST"&Scoring$ScoringALL==1,Scoring$Image,"X")
Scoring$CensorImageTimeDeadFirst<-ifelse(Scoring$ImageTimeDeadFirst=="X","",1)
w<-append(Scoring$ScoringALL,0,after=0)
Scoring$ScoringAllNminus1<-w[1:length(w)-1]
Scoring$ScoringAllnonFirst<-ifelse(Scoring$TimeAdjusted>0 & Scoring$ScoringALL==1 & Scoring$ScoringAllNminus1==0,Scoring$Image,"Y")
Scoring$CensorImageDeadExceptFirst<-ifelse(Scoring$ScoringAllnonFirst=="Y","",1)
Scoring$NotDeadLast<-ifelse(Scoring$CellPositionID_Last=="LAST" & Scoring$ScoringALL==0,Scoring$Image,"Z")
Scoring$CensorNotDeadLast<-ifelse(Scoring$NotDeadLast=="Z","",0)
Scoring$Well<-ifelse(Scoring$CensorImageDeadExceptFirst==1|Scoring$CensorImageTimeDeadFirst=="1"|Scoring$CensorNotDeadLast==0,Scoring$Well_ROI_ImagingPlate,"")
Scoring$Timepoint<-ifelse(Scoring$Well=="","",Scoring$TimeAdjusted)
Scoring$TimepointNumeric<-as.numeric(as.character(Scoring$Timepoint))
Scoring$Censorship<-ifelse(Scoring$Timepoint=="","",Scoring$ScoringALL)
Scoring$CensorshipNumeric<-as.numeric(as.character(Scoring$Censorship))
Scoring$ClassAndWellandImagingPlateInfo<-paste(Scoring$Image,Scoring$ImagingPlate,Scoring$Class, sep="_")
#write.csv(Scoring, paste(Experiment,"_scoring_output_with_Duplicates.csv",sep=""))
ScoringUnique<- Scoring[!(duplicated(Scoring$Well)|duplicated(Scoring$Well, fromLast = TRUE)),]
#write.csv(ScoringUnique, paste(Experiment,"_scoring_output_No_Duplicates.csv",sep=""))
ScoringUnique$well_noTimeZeroDead<-ifelse(ScoringUnique$TimepointNumeric==0,"X",ScoringUnique$Well)
ScoringUnique$Timepoint_noTimeZeroDead<-ifelse(ScoringUnique$TimepointNumeric==0,"",ScoringUnique$TimepointNumeric)
ScoringUnique$Censorship_noTimeZeroDead<-ifelse(ScoringUnique$TimepointNumeric==0,"",ScoringUnique$CensorshipNumeric)
ScoringUnique$Timepoint_noTimeZeroDead_Numeric<-as.numeric(as.character(ScoringUnique$Timepoint_noTimeZeroDead))
ScoringUnique$Censorship_noTimeZeroDead_Numeric<-as.numeric(as.character(ScoringUnique$Censorship_noTimeZeroDead))
ScoringUnique_NoTimeZeroDead_USE<- ScoringUnique[!(duplicated(ScoringUnique$well_noTimeZeroDead)|duplicated(ScoringUnique$well_noTimeZeroDead, fromLast = TRUE)),]
#write.csv(ScoringUnique_NoTimeZeroDead_USE, paste(Experiment,"_scoring_output_No_Duplicates_NoTimeZero.csv",sep=""))




#ScoringUnique_No_TimeZeroDead_USE: Can then be Read for survival output and HR using code below



ExperimentName<- ScoringUnique_NoTimeZeroDead_USE%>% group_by(Class)
groupsExamine2<-filter(ExperimentName, Class %in% groupsexamine)
ExperimentName2<-groupsExamine2
ExperimentName2$Class = factor(ExperimentName2$Class)
ExperimentName2$Class=relevel(ExperimentName2$Class,ref = Reference_Group)




newdata <- ExperimentName2

#write.csv(newdata, "newdata.csv")


#use 93 if want to include time zero dead
#sfit<-survfit(Surv(TimepointNumeric,CensorshipNumeric)~Class, data=newdata)

sfit<-survfit(Surv(Timepoint_noTimeZeroDead_Numeric,Censorship_noTimeZeroDead_Numeric)~Class, data=newdata)
SurvivalPlot<-ggsurvplot(sfit,palette = "grey",break.time.by=x_axisScale, conf.int=FALSE, pval=FALSE, risk.table=TRUE,legend.title="", main="Kaplan-Meier Curve",font.legend=10,legend=c("bottom"),font.x=15,font.y=15)
SurvivalPlot2<-SurvivalPlot+xlab(x_axisLabel)
#

jpeg(paste(Experiment,"_Survival_Curve.jpg", sep=""))
SurvivalPlot2
graphics.off()
#

CumulativeHazard <- ggsurvplot(sfit,palette = "grey",break.time.by=x_axisScale, pval=FALSE, fun="cumhaz", censor=FALSE,risk.table=FALSE,risk.table.title="Living Neurons",font.x=15,font.y=15,legend.title="",xlab="Day", font.legend=10,legend=c("top"))
CumulativeHazard2<-CumulativeHazard+xlab(x_axisLabel)
jpeg(paste(Experiment,"_Cumulative_Hazard.jpg", sep=""))
CumulativeHazard2
graphics.off()

#use 106 if with time zero
#fit <- coxph(Surv(TimepointNumeric, CensorshipNumeric)~Class, data=newdata)
fit <- coxph(Surv(Timepoint_noTimeZeroDead_Numeric,Censorship_noTimeZeroDead_Numeric)~Class, data=newdata)
COXPH_info<-summary(fit)
COXPH_HRoutput <- COXPH_info$coefficients
#write.csv(COXPH_HRoutput, paste(Experiment, "_Cumulative_Hazard_without_sfitSummary_noTime0dead.csv", sep= ""))
aaa<-COXPH_info$loglik
bbb<-COXPH_info$waldtest
ccc<-COXPH_info$rsq
ddd<-COXPH_info$concordance
eee<-COXPH_info$sctest
write.csv(eee,paste(Experiment, "_LogRankTest.csv", sep= "") )

#

sfit_summary <- summary(sfit)
sfit_summary_output<-sfit_summary$table
#write.csv(sfit_summary_output, paste(Experiment,"_Sfit_SummaryTable_Alone_notime0Dead.csv", sep= ""))
sfit_summary_output_mediary<-as.data.frame(sfit_summary_output)
hh<-append(sfit_summary_output_mediary$records[-1], "", length(sfit_summary_output_mediary$records-1))
yy<-hh[1:length(hh)-1]
oo<-append(sfit_summary_output_mediary$n.max[-1], "", length(sfit_summary_output_mediary$n.max-1))
pp<-oo[1:length(oo)-1]
aa<-append(sfit_summary_output_mediary$n.start[-1],"",length(sfit_summary_output_mediary$n.start-1))
bb<-aa[1:length(aa)-1]
cc<-append(sfit_summary_output_mediary$events[-1],"",length(sfit_summary_output_mediary$events-1))
dd<-cc[1:length(cc)-1]
ee<-append(sfit_summary_output_mediary$`*rmean`[-1],"",length(sfit_summary_output_mediary$`*rmean`-1))
ff<-ee[1:length(ee)-1]
gg<-append(sfit_summary_output_mediary$`*se(rmean)`[-1],"",length(sfit_summary_output_mediary$`*se(rmean)`-1))
hh<-gg[1:length(hh)-1]
ii<-append(sfit_summary_output_mediary$median[-1],"",length(sfit_summary_output_mediary$median-1))
jj<-ii[1:length(ii)-1]
kk<-append(sfit_summary_output_mediary$`0.95LCL`[-1],"",length(sfit_summary_output_mediary$`0.95LCL`-1))
ll<-kk[1:length(kk)-1]
mm<-append(sfit_summary_output_mediary$`0.95UCL`[-1],"",length(sfit_summary_output_mediary$`0.95UCL`-1))
nn<-mm[1:length(mm)-1]
COXPH_HRoutputWithN<- as.data.frame(COXPH_HRoutput)
COXPH_HRoutputWithN$records<- yy
COXPH_HRoutputWithN$n.max<-pp
COXPH_HRoutputWithN$n.start<-bb
COXPH_HRoutputWithN$events<-dd
COXPH_HRoutputWithN$'*rmean'<-ff
COXPH_HRoutputWithN$'*se(rmean)'<- hh
COXPH_HRoutputWithN$median<-jj
COXPH_HRoutputWithN$'0.95LCL'<-ll
COXPH_HRoutputWithN$'0.95UCL'<-nn
reference<- sfit_summary_output_mediary[1,]
referenceWithPvalue<-add_column(reference, "Pr(>|z|)"="", .before = 1)
referenceWithZ<-add_column(referenceWithPvalue, "z"="", .before = 1)
referenceWithSEcof<-add_column(referenceWithZ, "se(coef)"="", .before = 1)
referenceWithExpCoF<-add_column(referenceWithSEcof, "exp(coef)"=1, .before = 1)
referenceWithCoF<-add_column(referenceWithExpCoF, "coef"= "REFERENCE", .before=1)
COXPH_HRoutputWithN_USE<-rbind(referenceWithCoF,COXPH_HRoutputWithN)

write.csv(COXPH_HRoutputWithN_USE, paste(Experiment,"_Cumulative_Hazard_with_sfitSummaryTable.csv", sep = ""))
