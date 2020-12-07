/*
 * This file reads in images from path. The images in path should have rectangular regions selecting regions of interest from training
 * e.g. cell bodies.It doesn't matter how many rectangeles. Before saving the images the regions should have been converted to an 
 * overlay
 * 
 * This script reads in those images, and generates .txt files that are used by the train.py script to train the YOLO algorithm
 * The images should be placed in one directory
 * 
 * Currently the .txt files need to be changed to .xml but I will modify the python script so it can read in .txt.
 * 
 * I am also going to create a python script for annotation purposes also
 */
path=getDirectory("Choose Directory Containing training images");
list=getFileList(path);
outpath=getDirectory("Choose Directory for annotations");
x=0;
y=0;
width=0;
height=0;
outFile="";
setBatchMode(true);
for(k = 0; k < list.length; k++)
{
	name = split(list[k],".");
	outFile = File.open(outpath + name[0] + ".txt");

print(outFile,"<annotation verified=\"yes\">");
print(outFile,"	<folder>images</folder>");
print(outFile,"	<filename>"+list[k]+"</filename>");
print(outFile,"	<path>none</path>");
print(outFile,"	<source>");
print(outFile,"		<database>Unknown</database>");
print(outFile,"	</source>");
print(outFile,"	<size>");
print(outFile,"		<width>416</width>");
print(outFile,"		<height>416</height>");
print(outFile,"		<depth>1</depth>");
print(outFile,"	</size>");
print(outFile,"	<segmented>0</segmented>");
	
	roiManager("reset");
	open(path + list[k]);
	info = getInfo("overlay");
	if(info != "")
		run("To ROI Manager");
	n = roiManager("count");
	for(j = 0; j < n; j++)
	{
		roiManager("select", j);
		Roi.getBounds(x, y, width, height);
		xmin = x;
		xmax = x + width;
		ymin = y;
		ymax = y + height;
		if(xmin >= 0 && xmax <= 416 && ymin >= 0 && ymax <= 416)
		{
			print(outFile,"	<object>");
			print(outFile,"		<name>cells</name>");
			print(outFile,"		<pose>Unspecified</pose>");
			print(outFile,"		<truncated>0</truncated>");
			print(outFile,"		<difficult>0</difficult>");
			print(outFile,"		<bndbox>");
			print(outFile,"			<xmin>"+xmin+"</xmin>");
			print(outFile,"			<ymin>"+ymin+"</ymin>");
			print(outFile,"			<xmax>"+xmax+"</xmax>");
			print(outFile,"			<ymax>"+ymax+"</ymax>");
			print(outFile,"		</bndbox>");
			print(outFile,"	</object>");
		}
	}	
	print(outFile,"</annotation>");
	File.close(outFile);
	close();
	

}
