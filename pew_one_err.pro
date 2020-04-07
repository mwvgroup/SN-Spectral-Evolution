pro pew_one_err,w,f,ef,label,label2,xrg,yrg,x1man=x1man,x2man=x2man

if label2 eq 'CaHK' then restframe=3945.02
if label2 eq '4130' then restframe=4129.78 
if label2 eq 'MgII' then restframe=4481.00
if label2 eq 'FeII' then restframe=5169.00
if label2 eq 'SIIW1' then restframe=5449.20
if label2 eq 'SIIW2' then restframe=5622.46
if label2 eq '5972' then restframe=5971.89
if label2 eq '6355' then restframe=6356.08
if label2 eq 'CaII'  then restframe=8578.79 ;8578.79   ; 8498, 8542 and 8662
if label2 eq 'NaD' then restframe=5895.0
if label2 eq 'unk1' then restframe=7000.0
if label2 eq 'unk2'  then restframe=8000.0
if label2 eq 'FeI'   then restframe=7000.0
if label2 eq 'OI'    then restframe=7774.0
if label2 eq 'CaNIR' then restframe=(11784.+11839.+11849.)/3.

rang=[xrg,yrg]

plot,w,f,xr=rang,/ynozero
oplot,w,f+ef,col=cgcolor('grey')
oplot,w,f-ef,col=cgcolor('grey')
oplot,w,f
oplot,[restframe,restframe],[-100,100]

ok=0

while ok eq 0 do begin

  ;ch=get_kbrd()


  if keyword_set(x1man) then begin
    X1=x1man
    X2=x2man
  endif else begin
    ch=get_kbrd()
    CURSOR, X1, Y1, 0,/data
    oplot,[X1],[Y1],psym=1
    oplot,[X1,X1],[0.0,1.0],linestyle=2
    ;print,X1,Y1
    
    ch=get_kbrd()
    CURSOR, X2, Y2, 0,/data
    oplot,[X2],[Y2],psym=1
    oplot,[X2,X2],[0.0,1.0],linestyle=2
    ;print,X2,Y2
  endelse
  
  inl=where(w lt X1,npl)
  inh=where(w lt X2,nph)
  
  inin=where(w gt X1 and w lt X2, numpoints)
  ;print,w[inin]
  ;print,numpoints
  if X1 eq X2 then begin
    print,label,label2,'NO DATA'
    return
  endif
  if numpoints gt 5 then ok=1 else begin
    print,'RANGE TOO SMALL. PLEASE SELECT A WIDER RANGE'
    ok=0
  endelse
endwhile

d=fltarr(100)
ed=d
vel=d
evel=d
ew=d
eew=d
calc_area=d
earea=d
count=0
inl=inl-5
inh=inh-5
npl=npl-5
nph=nph-5

  for i=0,9 do begin
    for j=0,9 do begin

      X1=w[npl+i]
      X2=w[nph+j]
      Y1=f[npl+i]
      Y2=f[nph+j]
      Y=[Y1,Y2]
      X=[X1,X2]

;      oplot,X,Y,col=cgcolor('red')
      nw=w[npl+i:nph+j-1]
      nf=f[npl+i:nph+j-1]
      
      if n_elements(nf) eq 1 then begin
       print,'NO ENOUGH DATA TO MEASURE THE PARAMETERS'
       return
      endif

      bigarea=(X2-X1)*min(Y)+(X2-X1)*(max(Y)-min(Y))/2.         ;+((w[npl]-X1)*f[npl]+(X2-w[nph-1])*f[nph-1])

      smallarea=0.0
      for k=0,(nph+j-npl-i)-2 do begin
        yhere=[nf[k+1],nf[k]]
        smallarea=smallarea+((nw[k+1]-nw[k])*min(yhere)+(nw[k+1]-nw[k])*(max(yhere)-min(yhere))/2.)
      endfor
      ;;;AREA;;;
      calc_area[count]=bigarea-smallarea
      ;earea[count]
      
      nwb=X1+float(indgen(n_elements(nw)+1))*(X2-X1)/n_elements(nw)
      ywb=Y2+(Y1-Y2)/(X1-X2)*(nwb-X2)
      tabinv,nwb,nw,leff
      ywb=bspline_interpol(ywb,2,leff)

      ;ch=get_kbrd()

      ynorm=nf/ywb
;      plot,nw,ynorm,/ynozero,yr=[min(ynorm)-0.05,1.05]
;      oplot,[X1,X1],[0.0,2.0],linestyle=2
;      oplot,[X2,X2],[0.0,2.0],linestyle=2
;      oplot,[nw[0]-100.,nw[-1]+100],[1.0,1.0],linestyle=1,col=cgcolor('blue')

      pew=0.0
      for k=0,n_elements(nw)-2 do begin
        yhere=[ynorm[k+1],ynorm[k]]
        pew=pew+((nw[k+1]-nw[k])*min(yhere)+(nw[k+1]-nw[k])*(max(yhere)-min(yhere))/2.)
      endfor
      ;;;PASEUDOEW;;;
      ew[count]=(nw[n_elements(nw)-1]-nw[0])-pew
      ;eew[count]

      ;ch=get_kbrd()

      expr = 'gaussian(X, P[0:3])'
      start = [0.5,(X1+X2)/2.,50.,0.]
      result = MPFITEXPR(expr,nw,-(ynorm-1), rer,start,quiet=1)
      X3=result[1]
      oplot,[result[1],result[1]],[-100,100],col=cgcolor('red')
      inl2=where(nw lt X3,npl2)
      sy=size(ynorm,/dim)
      if sy gt npl2 then begin
        Y3=min([ynorm[npl2-2],ynorm[npl2-1],ynorm[npl2]]) 
        if Y3 eq ynorm[npl2-2] then npl2=npl2-1
        if Y3 eq ynorm[npl2] then npl2=npl2+1
      endif
      if sy eq npl2 then Y3=ynorm[npl2-1]
      if sy lt npl2 then begin
         Y3=ynorm[fix(sy/2.)]
         npl2=sy
      endif
      
      aqui=where(ynorm eq Y3)
      X=[nw[aqui]]
      X3=X[0]
;      oplot,[X3],[Y3],psym=1
;      oplot,[X3,X3],[Y3,1.0],col=cgcolor('red')
      ;;;ALTURA;;;
      d[count]=ywb[aqui]-nf[aqui]  
      ;ed[count]
      ;;;VELOCITAT;;;
      ;vel[count]=(restframe-result[1])/restframe*3e5/1000
      ;evel[count]
      ;doppler correction
      vel[count]=3e5*(((((restframe-result[1] )/restframe)+1)^2-1)/((((restframe-result[1] )/restframe)+1)^2+1))/1000.
      ;print,'old ',(restframe-result[1])/restframe*3e5/1000
      ;print,'new ',3e5*(((((restframe-result[1] )/restframe)+1)^2-1)/((((restframe-result[1] )/restframe)+1)^2+1))/1000.
      count++
    endfor
  endfor

;  plot,nw,ynorm,/ynozero,yr=[min(ynorm)-0.05,1.05]
;  oplot,nw, 1-gaussian(nw, result(0:3)), color=50, thick=5
;  oplot,[mean(vel)],[mean(vel)],psym=1
;  oplot,[mean(vel),mean(vel)],[Y3,1.0],col=cgcolor('red')
  print,label,label2,mean(vel),'(',stddev(vel),')',mean(ew),'(',stddev(ew),')',mean(d),'(',stddev(d),')',mean(calc_area),'(',stddev(calc_area),')',$
  format='(a10,2x,a10,2x,f7.2,a1,f5.2,a1,2x,f6.2,a1,f5.2,a1,2x,e9.2,a1,e9.2,a1,2x,e9.2,a1,e9.2,a1)'

  ;ch=get_kbrd()


END
