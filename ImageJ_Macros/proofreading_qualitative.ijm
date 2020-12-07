//Dialog.create("Chooser");
//Dialog.addString("Title:", "make a choice");
//Dialog.addChoice("Type:", newArray("Keep All","Keep Changed"));
//Dialog.addCheckbox("Make CSV", true)
//Dialog.show()
//type = Dialog.getChoice();
//makeCSV = Dialog.getCheckbox();
//print(type);

inDir = getDirectory("Choose Directory Containing Files ");
//Alive = getDirectory("Choose Directory Containing Alive ");
//Dead = getDirectory("Choose Directory Containing Dead ");
//resultDir = getDirectory("Choose Results Directory ");
files = getFileList(inDir);
n= files.length
n = n/2
files2 = newArray(n);
f2i = 0
for(k =0; k < files.length; k++)
{
	ext = split(files[k],'.');	
	if(ext[1] == "csv")
	{
		files2[f2i] = ext[0];
		f2i++;
	}
	
}
for(file_num = 0; file_num < files2.length; file_num++)
{
setBatchMode(false);	
open(inDir + files2[file_num]+'.csv');
open(inDir + files2[file_num]+'montage.tif');
n=nResults;
getDimensions(width,height,channels,slices,frames);
t = width / 100;
row = 0;
col = 0;
roiManager("reset");
ROI = newArray(n);
for(k = 0; k < n; k++)
{
s = getResultString("Status",k);
if(s == "True")
{
ROI[k] = true;
makeRectangle(col * 100,row*100,90,90);
roiManager("Add");
}
else
{
	ROI[k] = false;
}
col++;
if(col >= t)
	{
		col = 0;	
		row++;
	}
}

predicted = newArray(n);
for(k = 0; k < n; k++)
{
	if(ROI[k])
	predicted[k] = true;
	else
	predicted[k] = false;
}

print("stuff");
 shift=1;
      ctrl=2; 
      rightButton=4;
      alt=8;
      leftButton=16;
      insideROI = 32; // requires 1.42i or later

      x2=-1; y2=-1; z2=-1; flags2=-1;
      logOpened = false;
      if (getVersion>="1.37r")
          setOption("DisablePopupMenu", true);
      while (!logOpened || isOpen("Log")) {
          getCursorLoc(x, y, z, flags);
          if (x!=x2 || y!=y2 || z!=z2 || flags!=flags2) {
              s = " ";
              if (flags&leftButton!=0)
              {
              	val = t * floor(y / 100) + floor(x/100);                  
              	if(ROI[val])
              		 {
              		 	ROI[val] = false;
              		 }
              	else
              		{
              			ROI[val] = true;              			              			
              		}
              	roiManager("reset");
              	col = 0; row = 0;
              	for(k = 0; k < ROI.length; k++)
				{
					if(ROI[k])							
					{
						makeRectangle(col * 100,row*100,90,90);
						roiManager("Add");
					}
					col++;
					if(col >= t)
					{
						col = 0;	
						row++;
					}
				}
				wait(500);
              }
              if (flags&rightButton!=0)
              {
              	 val = t * floor(y / 100) + floor(x/100);
              	 col = floor(x/100);                  
              	if(ROI[val])
              		 {
              		 	for(j = 0; j < t-col; j++)
              		 		{
              		 			ROI[val+j] = false;
              		 		}
              		 }
              	else
              		{
              			for(j = 0; j < t-col; j++)
              		 	{
              		 		ROI[val+j] = true;              			              			
              		 	}
              		}
              	roiManager("reset");
              	col = 0; row = 0;
              	for(k = 0; k < ROI.length; k++)
				{
					if(ROI[k])							
					{
						makeRectangle(col * 100,row*100,90,90);
						roiManager("Add");
					}
					col++;
					if(col >= t)
					{
						col = 0;	
						row++;
					}
				}
				wait(500);          
              }
              logOpened = true;              
          }
          x2=x; y2=y; z2=z; flags2=flags;
          wait(10);
      }
      if (getVersion>="1.37r")
          setOption("DisablePopupMenu", false);

col = 0; row = 0;
roiManager("reset");
setBatchMode(true);
for(k = 0; k < ROI.length; k++)
{
	//if(type == "Keep All" || (type == "Keep Changed" && ROI[k] != predicted[k]))//will only save images classified incorrectly
	{
	makeRectangle(col * 100,row*100,100,100);
	title = getTitle();
	name = split(title,".");
	print(name[0]);
	//exit();		
	run("Duplicate...", " ");	
	//outDir = Dead;
	if(ROI[k])
		{
			//outDir = Alive;			
			
		}
		//saveAs("PNG", outDir + name[0] + "_" + k + 1 + ".png");
					
		
	
	
		
	close();
	}		
	
	col++;
	if(col >= t)
	{
		col = 0;	
		row++;
	}
}
AA=0;
DD=0;
AD=0;
DA=0;

for(k = 0; k < ROI.length; k++)
{
if(ROI[k])
{
	setResult("Manual Status",k,"TRUE");
}
else
{
	setResult("Manual Status",k,"FALSE");
}
if(predicted[k] && ROI[k])
AA++;
if(!predicted[k] && !ROI[k])
DD++;
if(predicted[k] && !ROI[k])
AD++;
if(!predicted[k] && ROI[k])
DA++;
}
setResult("AA",0,AA);
setResult("DD",0,DD);
setResult("AD",0,AD);
setResult("DA",0,DA);
//saveAs("Results", resultDir + files2[file_num]+'.csv');
print(AA + " " + DD + " " + AD + " " + DA);
close("Results");
close();
}
