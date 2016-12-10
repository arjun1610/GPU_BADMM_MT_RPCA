function convertDat2Avi(filename, saveFilename)

n = 200;
m = 25344;
fid = fopen(filename,'r');
X = fread(fid,[n,m],'float'); 
fclose(fid); 
X = X';

% above can be a problem as the output may not contain the matrix dimensions as 3 times no. of frames.

% call lrslibary.utils.convert_video2d_to_avi
convert_video2d_to_avi( X, 200, 144, 176, saveFilename);