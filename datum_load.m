XO = load( 'hall1-200.mat' );
XO_1 = XO.XO(:,1:200);

m = 25344;
n = 200;
C = XO_1;
fid = fopen([int2str(n) 'C.dat'],'w');
fwrite(fid,[m,n],'int');
fwrite(fid,C','float');
fclose(fid);
