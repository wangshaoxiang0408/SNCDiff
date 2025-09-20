import matplotlib.pyplot as plt
import numpy as np
import skill_metrics as sm
import pickle
#格式
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
import seaborn as sns


# plt.style.use(['science', 'no-latex'])
import warnings
warnings.filterwarnings('ignore') 
linewidth=3
alpha=0.8

font_prop_legend= FontProperties(family='Times New Roman')
fontdict_legend={'family': 'Times New Roman', 'size': 15}
# Set global font size
mpl.rcParams['font.size'] = 25
# Set global font size for labels
mpl.rcParams['axes.labelsize'] = 27
# Set global font size for tick labels
mpl.rcParams['xtick.labelsize'] = 27
mpl.rcParams['ytick.labelsize'] = 27
mpl.rcParams['legend.fontsize'] = 27

def save_obj(obj, name):
    # Save object to file in pickle format
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

class Container(object): 
    def __init__(self, sdev, crmsd, ccoef, gageID):
        self.sdev = sdev
        self.crmsd = crmsd
        self.ccoef = ccoef
        self.gageID = gageID


import csv

# 打开CSV文件
with open('0506/progress.csv', 'r') as csvfile:
    # 创建CSV阅读器对象
    csv_reader = csv.reader(csvfile)
    
    # 逐行读取CSV文件
    data = []
    cal = []
    diff = []
    cal_q0 = []
    cal_q1 = []
    cal_q2 = []
    cal_q3 = []
    diff_q0 = []
    diff_q1 = []
    diff_q2 = []
    diff_q3 = []
    grad_norm = []
    loss = []
    loss_q0 = []
    loss_q1 = []
    loss_q2 = []
    loss_q3 = []
    param_norm = []
    samples = []
    step = []
    
        
    for row in csv_reader:
        # data.append(row)
        # data.append(row)
        cal.append(row[0])
        diff.append(row[5])
        cal_q0.append(row[1])
        cal_q1.append(row[2])
        cal_q2.append(row[3])
        cal_q3.append(row[4])
        diff_q0 .append(row[6])
        diff_q1.append(row[7])
        diff_q2.append(row[8])
        diff_q3.append(row[9])
        grad_norm.append(row[10])
        loss.append(row[11])
        loss_q0.append(row[12])
        loss_q1.append(row[13])
        loss_q2.append(row[14])
        loss_q3.append(row[15])
        param_norm.append(row[16])
        samples.append(row[17])
        step.append(row[18])       

# 打印读取到的数据
cal = cal[1:]
diff = diff[1:]
cal_q0 = cal_q0[1:]
cal_q1 = cal_q1[1:]
cal_q2 = cal_q2[1:]
cal_q3 = cal_q3[1:]
diff_q0 = diff_q0[1:]
diff_q1 = diff_q1[1:]
diff_q2 = diff_q2[1:]
diff_q3 = diff_q3[1:]
grad_norm = grad_norm[1:]
loss = loss[1:]
loss_q0 = loss_q0[1:]
loss_q1 = loss_q1[1:]
loss_q2 = loss_q2[1:]
loss_q3 = loss_q3[1:]
param_norm = param_norm[1:]
samples = samples[1:]
step = step[1:]
cal = np.array(cal).astype(float)
diff = np.array(diff).astype(float)
cal_q0 = np.array(cal_q0).astype(float)
cal_q1 = np.array(cal_q1).astype(float)
cal_q2 = np.array(cal_q2).astype(float)
cal_q3 = np.array(cal_q3).astype(float)
diff_q0 = np.array(diff_q0).astype(float)
diff_q1 = np.array(diff_q1).astype(float)
diff_q2 = np.array(diff_q2).astype(float)
diff_q3 = np.array(diff_q3).astype(float)
grad_norm = np.array(grad_norm).astype(float)
loss = np.array(loss).astype(float)
loss_q0 = np.array(loss_q0).astype(float)
loss_q1 = np.array(loss_q1).astype(float)
loss_q2 = np.array(loss_q2).astype(float)
loss_q3 = np.array(loss_q3).astype(float)
param_norm = np.array(param_norm).astype(float)
samples = np.array(samples).astype(float)
step = np.array(step).astype(float)

def stage (cal,diff):
    step_cal = []
    step_diff = []
    len_c = len (cal)
    # step_cal.append(cal[0:7096])
    # step_diff.append(diff[0:7096])
    for i in range(1,len_c):
        if i%4000==0:
            step_cal.append(cal[i+20:i+2000])
            step_diff.append(diff[i+20:i+2000])
    return step_cal,step_diff
cal_step, diff_step = stage(cal,diff)    
import os
import time
#-----------------------------------------画小提琴图--------------------------------------
# cal_step, diff_step = stage(cal,diff)
# sns.violinplot(x=None, y=None, hue=None, data=cal_step, order=None, hue_order=None,
#         bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100,
#         width=0.8, inner='box', split=False, dodge=True, orient=None,
#         linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
# sns.violinplot(x=None, y=None, hue=None, data=diff_step, order=None, hue_order=None,
#         bw='scott', cut=2, scale='area', scale_hue=True, gridsize=100,
#         width=0.8, inner='box', split=False, dodge=True, orient=None,
#         linewidth=None, color=None, palette=None, saturation=0.75, ax=None)
# plt.show()
# SaveDir = 'result/'
# if not os.path.exists(SaveDir):
#     os.makedirs(SaveDir)
# now = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(
#     time.time()))  # 获得当前时间 2021_1108_2310_22
# SaveFile = SaveDir + 'zzz' + '_' + now + '.png'  
# plt.savefig(SaveFile)
# print(SaveFile +    ' ==================>>> has been complete')
# plt.close()

#-----------------------------------------结束画小提琴图--------------------------------------
#-----------------------------------------画泰勒图--------------------------------------
# 生成一些虚拟数据

num_models =8
np.random.seed(0)
sdev = np.array([1.0] + list(np.random.uniform(0.5, 1.5, num_models)))
crmsd = np.random.uniform(0.1, 0.5, num_models + 1)
ccoef = np.array([1.0] + list(np.random.uniform(0.5, 1.0, num_models)))
gageID = ['Ref'] + ['Model {}'.format(i) for i in range(1, num_models + 1)]
# 将数据存储在Container对象中，并保存到pickle文件
data = Container(sdev, crmsd, ccoef, gageID)
# 将数据存储在Container对象中，并保存到pickle文件
data = Container(sdev,crmsd,ccoef,gageID )
save_obj(data, 'Farmington_River_data')
# 从pickle文件读取数据
data = pickle.load(open('Farmington_River_data.pkl', 'rb'))

# Close any previously open graphics windows
plt.close('all')
# Plot with a large number of symbols of different color along with a legend.
plt.figure(figsize=(12, 10))
# Produce the Taylor diagram
# All the input arrays must have the same length
sm.taylor_diagram(data.sdev, data.crmsd, data.ccoef, 
                  numberPanels=1, 
                  markerLabel=data.gageID, 
                  markerLegend='on', 
                  markerColor='k', markerSize=20, 
                  alpha=0.7, 
                  #tickRMS=np.arange(0,1.8,0.2), 
                  tickRMS = [0, 0.2,0.4, 0.5, 1],
                  tickRMSangle=130.0,
                  tickCOR=np.arange(0.4,1.05,0.1),
                  showlabelsRMS='on', 
                  titleRMS='off', 
                  titleOBS = 'Observation',colOBS='r',markerobs='o',styleOBS='-',)


ax = plt.gca()
# 生成泰勒图后，获取所有文本对象
text_objects = [child for child in ax.get_children() if isinstance(child, mpl.text.Text)]

# 遍历文本对象并设置样式
for text_obj in text_objects:
    if 'Correlation' in text_obj.get_text():
        text_obj.set_family('Times New Roman')  # 设置字体
        #text_obj.set_size(16)                  # 设置字体大小
        text_obj.set_weight('bold')            # 设置字体为粗体


# 设置坐标轴的刻度格式
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.text(0.5, 0.2, 'CRMSD',transform=ax.transAxes, 
            verticalalignment='top',c='b',fontweight='bold')
ax.set_ylabel('Normalized Standard Deviation(NSD)',c='m',weight='bold',)

plt.suptitle('Result Tayler Plot 1',weight='bold',y=0.99)
plt.tight_layout()
plt.savefig(r'taylor1.png')
#plt.show()



# Specify labels for points in a cell array
# label = ['Reference', 'Model A', 'Model B', 'Model C', 'Model D']
bias = np.random.uniform(-20, 20, num_models + 1)

# 生成第二个泰勒图
plt.figure(figsize=(12, 10))
sm.taylor_diagram(sdev, crmsd, ccoef, markerLabel=data.gageID,
                  locationColorBar='EastOutside', markerDisplayed='colorBar', 
                  titleColorBar='Bias', markerLabelColor='black', markerSize=20,
                  markerLegend='off', cmapzdata=bias, colRMS='g', styleRMS=':', 
                  widthRMS=2.0, titleRMS='off', colSTD='b', styleSTD='-.', 
                  widthSTD=1.0, titleSTD='on', colCOR='k', styleCOR='--', 
                  widthCOR=1.0, titleCOR='on',
                  titleOBS = 'Observation',colOBS='r',markerobs='o',styleOBS='-',)


# 生成泰勒图后，获取所有文本对象
text_objects = [child for child in ax.get_children() if isinstance(child, mpl.text.Text)]
# 遍历文本对象并设置样式
for text_obj in text_objects:
    if 'Correlation' in text_obj.get_text():
        text_obj.set_family('Times New Roman')  # 设置字体
        #text_obj.set_size(27)                  # 设置字体大小
        text_obj.set_weight('bold')            # 设置字体为粗体

# 设置坐标轴的刻度格式
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax.text(0.5, 0.2, 'CRMSD',transform=ax.transAxes, 
            verticalalignment='top',c='b',fontweight='bold')
ax.set_ylabel('Normalized Standard Deviation(NSD)',c='m',weight='bold',)
plt.suptitle('Result Tayler Plot 2',weight='bold',y=0.99)
plt.tight_layout()
plt.savefig(r'taylor2.png')
#plt.show()