dir1 = getDirectory("Choose Source Directory ");
dir2 = getDirectory("Choose Destination Directory ");

list = getFileList(dir1);
setBatchMode(true);


for (k=0; k<list.length; k++) {

showProgress(k+1, list.length);




open(dir1+list[k]);

title = getTitle();
//print(title);



run("Subtract Background...", "rolling=40 stack");



saveAs("Tiff", dir2+title);
close();

}