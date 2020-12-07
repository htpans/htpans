//in_path = "C:/Users/Eric Danielson/Desktop/yolo_stitched/"
//out_path = "C:/Users/Eric Danielson/Desktop/yolo_temp/";
in_path = getDirectory("Choose Directory Containing Stitched Images with overlay");
out_path= getDirectory("Choose Directory to Write Files ");
yInc=416;
xInc=416;

setBatchMode(true);
image_list=getFileList(in_path);

for(image_index=0;image_index < image_list.length; image_index++)
{
roiManager("reset");
open(in_path + image_list[image_index]);
run("To ROI Manager"); //regions

width=0;
height=0;
channels=0;
slices=0;
frames=0;
getDimensions(width, height, channels, slices, frames);

region_number=roiManager("count");
xList=newArray(region_number);
yList=newArray(region_number);
widthList=newArray(region_number);
heightList=newArray(region_number);
onEdge =newArray(region_number);

for(r=0;r<region_number;r++)
{
	roiManager("select",r);	
	Roi.getBounds(xList[r], yList[r], widthList[r], heightList[r]);	
	onEdge[r]=false;
}

run("From ROI Manager"); //put overlay back
title=getTitle();
root = split(title, ".");
title = root[0];
image_num=1;
//saves all imags to yolo_train folder with overlay ontop
for(y=0; y < height; y = y + yInc)
{
	for(x=0; x < width; x = x+ xInc)
	{
		x2=x;
		y2=y;
		if(x + xInc > width)
			x2 = width- xInc;
		if(y + yInc > height)
			y2 = height - yInc;
		makeRectangle(x2, y2, xInc, yInc);
		run("Duplicate...", "title ="+title + "_" + image_num);
		saveAs("Tiff", out_path + title + "_" + image_num);
		close();
		image_num++;
	}
}
//saves extra images where region was on edge
//print("here");
for(y = 0; y < height; y = y + yInc)
{
	for(r=0; r< region_number; r++)
	{
		if(y >= yList[r] && y <= yList[r]+heightList[r] )
			onEdge[r]=true;
	}
}

for(x=0; x < width; x= x+ xInc)
{
	for(r=0; r< region_number; r++)
	{
		if(x >= xList[r] && x <= xList[r]+widthList[r] )
			onEdge[r]=true;
	}
}

for(index=0; index < region_number; index++)
{
	
	if(onEdge[index])
	{
		roiManager("select", index);
		//print(index);
		x2 = xList[index] - xInc/2;
		y2 = yList[index] - yInc/2;
		if(x2 < 0)
			x2=0;
		if(y2 < 0)
			y2 =0;
		makeRectangle(x2, y2, xInc, yInc);
		run("Duplicate...", "title ="+title + "_" + image_num);
		saveAs("Tiff", out_path + title + "_" + image_num);
		close();
		//print(title + "_" + image_num);
		image_num++;
	}
		
}
close();
}