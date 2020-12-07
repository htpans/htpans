in_path = getDirectory("Choose Directory Containing yolo_training images");
image_list=getFileList(in_path);
for(image_index=0;image_index < image_list.length; image_index++)
{
roiManager("reset");
open(in_path + image_list[image_index]);
info = getInfo("overlay");
if(info != "")
	run("To ROI Manager");
waitForUser("Inspect Image", "Check for additional cells, select with rectangle tool, add with 'T',click 'OK' when finished ");
n=roiManager("count");
if(n>0)
	run("From ROI Manager");
saveAs("Tiff",in_path + image_list[image_index]);
close();
}
