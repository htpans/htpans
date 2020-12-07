requires("1.39u");
inDir = getDirectory("Choose Directory Containing Files ");
outDir= getDirectory("Choose Directory to Write Files ");

list = getFileList(inDir);
setBatchMode(true);
 //wells = 4; 
 //channel = newArray("RFP");

 //Calculate well number
 wellList = newArray(384);
 l=0;
num=0;
rows = newArray("A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P");
cols = newArray("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24");
row = 0;
col = 0;

for(k = 0; k < list.length; k++)
{
l = substring(list[k],0,1);
num = substring(list[k],1,3);	
if(substring(list[k],2,3) == "_") //num < 10
num = substring(list[k],1,2);	

col = 0;
while(num != cols[col])
{
	col++;	
}
row=0;
while(l != rows[row])
{
	row++;	
}
wellList[col + (row*24)] = true;
}
wellNumber = 0;
for(k = 0; k < wellList.length; k++)
{
	if(wellList[k])
		wellNumber++;	
}
wells = wellNumber;
print(wellNumber);
//END calculate well number

//calculate channel number
channelList = newArray("0","0","0","0");

for(k = 0; k < list.length; k++)
{
ns = indexOf(list[k],"[");
ne = indexOf(list[k],"]");
s = substring(list[k],ns,ne+1);
cC = 0;
while(s != channelList[cC])
{
	if(channelList[cC] == "0")
	channelList[cC] = s;
	else
	cC++;		
}

}
channelNum = 0;
for(k = 0; k < channelList.length; k++)
{
	if(channelList[k] != "0")
	channelNum++;
}

channel = newArray(channelNum);

for(k = 0; k < channelNum; k++)
{
channel[k] = channelList[k];
}
//end calculate channel number
 
 images_per_timepoint = wells * channel.length;
 timepoints = list.length / images_per_timepoint;
 
for(imageNum = 0; imageNum < list.length; imageNum++)
{
open(inDir + list[imageNum]);
//print(list[imageNum]);
s = getMetadata("Info");
ds = indexOf(s,"<Date>");
de = indexOf(s,"</Date>");
ts = indexOf(s,"<Time>");
te = indexOf(s,"</Time>");
date=substring(s,ds+6,de);
olddate=date;
newdate = split(date,'/');
date = newdate[2] + "/" + newdate[0] + "/" + newdate[1];
time=substring(s,ts+6,te);
ns = date + ":" + time;
//print(olddate);
//print(date);

close(list[imageNum]);
list[imageNum] = ns + list[imageNum];

}
Array.sort(list);
for(imageNum = 0; imageNum < list.length; imageNum++)
{
list[imageNum] = substring(list[imageNum],17,lengthOf(list[imageNum]));	
//print(list[imageNum]);
}



for(w = 0; w < wells; w++)
{

for(ch = 0; ch < channel.length; ch++)
{
nameRoot=split(list[w + ch],"_");
base=nameRoot[0];
//outFile = outDir + base + channel[ch] + ".tif";
outFile = outDir + base + ".tif";
for (t=0; t < timepoints ;t++) 
{
open(inDir+list[w + (images_per_timepoint*t)]);
}
run("Images to Stack", "method=[Copy (center)] name=Stack title=[] use");
saveAs("Tiff", outFile);
close();

//end timepoint


} //end channel

} //end wells
print("Done");