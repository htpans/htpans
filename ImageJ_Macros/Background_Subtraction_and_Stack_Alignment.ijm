dir1 = getDirectory("Choose Source Directory ");
dir2 = getDirectory("Choose Destination Directory ");

list = getFileList(dir1);
setBatchMode(true);


for (k=0; k<list.length; k++) {

showProgress(k+1, list.length);




open(dir1+list[k]);

title = getTitle();
//print(title);


run("Duplicate...", "title=Stack.tif duplicate");
selectWindow(title);
run("Close");
selectWindow("Stack.tif");

run("Subtract Background...", "rolling=40 stack");

selectWindow("Stack.tif");
setSlice(1);

run("MultiStackReg", "stack_1=Stack.tif action_1=Align file_1=[] stack_2=None action_2=Ignore file_2=[] transformation=[Rigid Body]");

selectWindow("Stack.tif");

rename(title);

saveAs("Tiff", dir2+title);
close();

}