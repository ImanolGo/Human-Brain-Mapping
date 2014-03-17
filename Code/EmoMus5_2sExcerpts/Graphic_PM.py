from xlrd import open_workbook
import numpy as np
from pylab import *


wb = open_workbook('ResultsPM_OneModel.xls')
results = []
users = []

for s in wb.sheets():
	values = []
	users.append(s.name)
	for col in range(1,s.ncols):
		str = s.cell(s.nrows-1,col).value
		values.append( float(str.split("%")[0]))
	results.append(values)

results = np.array(results)
	

	

t = arange(1, len(results[:,0])+1, 1)
Linear_Reg = results[:,0]
SVR_poly = results[:,1]
SVR_rbf  = results[:,2]

p1 = plot (t,Linear_Reg, 'b',linewidth=2 )
plot (t,Linear_Reg,'bo')
p2 = plot(t,SVR_poly, 'k' ,linewidth=2)
plot (t,SVR_poly,'ko')
p3 = plot(t,SVR_rbf, 'g' ,linewidth=2)
plot (t,SVR_rbf,'go')


legend( (p1,p2,p3), ('Linear_Reg', 'SVR_poly','SVR_rbf'), 'center right', shadow=True)
xtext = xlabel('Users')
ytext = ylabel('Prediction Accuraccy [%]')
ttext = title('Prediction Model')
axis([1, len(results[:,0]),0,100])
grid(True)


setp(ttext, size='large', name='helvetica', weight='bold')
setp(xtext, size='large', name='helvetica', weight='bold')
setp(ytext, size='large', name='helvetica', weight='bold')

show()


