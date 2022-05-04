import time
import random
import datetime
import requests
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit.server.server import Server
from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
from st_aggrid import AgGrid
from pyecharts.charts import *
from pyecharts import options as opts
from pyecharts.globals import SymbolType
import plotly.figure_factory as ff
from lxml import etree
import os
import plotly.express as px
from jieba import lcut, load_userdict, add_word
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


name2id = {'北京': 1, '上海': 3, '广州': 48, '佛山': 53, '杭州': 37, '海口': 122}

def get_city_price(city, year):
    '''获取城市房价信息'''
    global tmp
    name2id = {'北京': 1, '上海': 3, '广州': 48, '佛山': 53, '杭州': 37, '海口': 122}
    file_name = f'./city_data/{city}_{year}.csv'
    if year == 2022:
        if not os.path.exists(file_name):
            data2 = pd.read_html(f'https://fangjia.gotohui.com/fjdata-{name2id[city]}')
            print('正在获取网络数据……\n', f'https://fangjia.gotohui.com/fjdata-{name2id[city]}')
            tmp = data2[1]
            tmp = tmp.loc[tmp['日期'].apply(lambda x: x in ['2022-03', '2022-02', '2022-01']), ['日期', '二手房(元/㎡)', '新房(元/㎡)']]
            tmp['city'] = city
            tmp.columns = ['月份', '二手房均价', '新房均价', '城市']
            tmp['月份'] = tmp['月份'].apply(lambda x: f'{x.split("-")[0]}年{x.split("-")[1]}月')
            tmp['新房均价'] = tmp['新房均价'].apply(lambda x: str(x)+'元/㎡')
            tmp['二手房均价'] = tmp['二手房均价'].apply(lambda x: str(x) + '元/㎡')
            tmp.to_csv(file_name)
            print('数据获取完毕。')
        else:
            print('已有数据，数据读取中……')
            tmp = pd.read_csv(file_name, index_col=0)
            print('读取完毕。')
    else:
        if not os.path.exists(file_name):
            url = f'https://fangjia.gotohui.com/years/{name2id[city]}/{year}/'
            print('正在获取网络数据……\n', url)
            data1 = pd.read_html(url)
            tmp = pd.DataFrame(data1[0].values[1:], columns=data1[0].values[0])
            tmp['月份'] = tmp['月份'].apply(lambda x: f'{year}年{x}')
            tmp['城市'] = city
            tmp.to_csv(file_name)
            print('数据获取完毕。')
        else:
            print('已有数据，数据读取中……')
            tmp = pd.read_csv(file_name, index_col=0)
            print('读取完毕。')
    return tmp

def my_str(string):
    import re
    tmp = re.search('([0-9]{4}).0?([0-9]{1,2})', string).groups()
    return f'{tmp[0]}-{tmp[1]:0>2}'

def get_gz_house(nums=10, file_name='gzdata_all.csv'):
    def GetHeader():
        with open('user-agents.txt', 'r') as fhand:
            agent = random.choice(fhand.read().split('\n'))
        header = {
            'User-Agent': agent,
            'referer': 'https://www.anjuke.com/'
        }
        return header

    def my_process(my_str):
        t = my_str.split(' | ')
        # print(t)
        if len(t) == 7:
            return t
        elif len(t) == 6:
            return t[:-1] + [''] + t[-1:]
        elif len(t) > 7:
            return t[:7]

    if not os.path.exists(file_name):
        names, locations, total_prices, unit_prices, house_infos = [], [], [], [], []
        # 2循环爬取
        for i in range(1, nums + 1):
            print('正在获取第{}页数据'.format(i))
            url = f'https://gz.lianjia.com/ershoufang/pg{i}/'
            web_data = requests.get(url, headers=GetHeader())  # 发送HTTP请求
            dom = etree.HTML(web_data.text)
            name = dom.xpath('//*[@id="content"]/div[1]/ul/li/div[1]/div[1]/a/text()')
            location = dom.xpath('//div[@class="positionInfo"]')
            location = [i.xpath('string(.)') for i in location]  # 取下面的所有文字
            total_price = dom.xpath('//div[@class="totalPrice totalPrice2"]/span/text()')
            unit_price = dom.xpath('//div[@class="unitPrice"]/span/text()')
            house_info = dom.xpath('//div[@class="houseInfo"]/text()')

            # 将中间数据保存下来
            names.extend(name)
            locations.extend(location)
            total_prices.extend(total_price)
            unit_prices.extend(unit_price)
            house_infos.extend(house_info)
            time.sleep(2)

        tmp = [my_process(i) for i in house_infos]
        data_all = pd.DataFrame(
            tmp, columns=['房屋户型', '房屋套内面积', '房屋朝向', '装修情况', '所在楼层', '建筑时间', '建筑类型']
        )
        data_all['房屋标题'] = names
        data_all['房屋所在位置'] = locations
        data_all['房屋总价'] = total_prices
        data_all['房屋单价'] = unit_prices
        data_all['采集时间'] = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        data_all.to_csv(file_name, encoding='gbk')
        return data_all
    else:
        data_all = pd.read_csv(file_name, index_col=0, encoding='gbk')
        return data_all

def process_gz(data):
    import re
    data['房屋套内面积'] = data['房屋套内面积'].str.replace('平米', '').astype('float')
    data['房屋单价'] = data['房屋单价'].str.replace('元/平|,', '').astype('float')
    data['厅数量'] = data['房屋户型'].apply(lambda x: int(re.findall('([0-9]+)厅', x)[0]))
    data['房间数量'] = data['房屋户型'].apply(lambda x: int(re.findall('([0-9]+)室', x)[0]))

    def get_buliding_year(my_str):
        t = re.findall('([0-9]+)年建', my_str)
        if len(t) == 0:
            return None
        else:
            return 2022 - int(t[0])

    data['楼龄'] = data['建筑时间'].astype(str).apply(get_buliding_year)
    data['楼龄'] = data['楼龄'].fillna(int(data['楼龄'].mean()))
    return data

def get_freq(data):
    # 可视化绘图
    for i in set(_flatten(list(data['房屋所在位置'].str.split('    -  ')))):
        add_word(i)
    for i in set(data['房屋户型']):
        add_word(i)
    add_word('望花园')
    add_word('刚需')
    load_userdict('./guangzhou.txt')
    t = [' '.join(i) for i in data['房屋标题'].apply(lcut)]
    t = ' '.join(t)
    num = pd.Series(t.split(' ')).value_counts()
    freq = num[[i for i in num.index if i not in [' ', '，', '。']]]
    return freq

# %% ----------网页设置----------
st.set_page_config(
    page_title="热门城市房价采集及分析", page_icon=":rainbow:",
    layout='wide', initial_sidebar_state="auto",
)

st.sidebar.title('虚拟仿真实验')
st.sidebar.button('房地产数据')
st.sidebar.button('环境数据')
st.sidebar.button('消费数据')
session_id = get_report_ctx().session_id
sessions = Server.get_current()._session_info_by_id
session_ws = sessions[session_id].ws
st.sidebar.info(f'当前在线人数：{len(sessions)}')

# %% ----------标题----------
st.title(':heart:热门城市房价采集及分析:heart:')
st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html=True)

# %% ----------会话状态----------
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
else:
    st.session_state.first_visit = False
# 初始化全局配置
if st.session_state.first_visit:
    st.session_state.date_time = datetime.datetime.now() + datetime.timedelta(hours=8)  # Streamlit Cloud的时区是UTC，加8小时即北京时间
    st.balloons()
    st.snow()

st.write('# :star:热门城市数据采集与分析')
# ----------参数面板----------
st.write('## :dizzy:数据采集')
citys = st.multiselect('选择爬取的城市：👇', ['北京', '上海',  '广州', '佛山', '杭州', '海口'])
years = st.multiselect('选择爬取的年份：👇', [2019, 2020, 2021, 2022])
if st.button('开始爬取'):
    all_data = pd.DataFrame([])
    for city in citys:
        for year in years:
            tmp = get_city_price(city, year)
            all_data = pd.concat([all_data, tmp], axis=0)
            time.sleep(1)

    with st.expander('是否查看采集得到的数据？'):
        with st.form('example form') as f:
            ag = AgGrid(
                all_data,
                height=400,
                fit_columns_on_grid_load=True,
                reload_data=False
            )
            st.form_submit_button()

st.write('## :dizzy:是否查看所有城市的二手房房价数据？')
with st.expander('是'):
    data = pd.read_excel('./city_data/house_price_city.xlsx', index_col=0)
    data['月份'] = data['月份'].apply(my_str)
    data2 = pd.pivot_table(data, index='月份', columns='城市', values='二手房均价')
    st.dataframe(data2)

st.write('## :dizzy:描述性统计')
with st.expander('点击查看'):
    tmp = data2.describe()
    st.dataframe(tmp)

st.write('## :dizzy:数据可视化')
with st.expander('点击查看'):
    line = Line()
    line.add_xaxis(list(data2.index))
    for city in name2id.keys():
        data_city = data2[city]
        line.add_yaxis(
            city, y_axis=list(data_city),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最大值"), opts.MarkPointItem(type_="min", name="最小值"),
                ]
            ),
        )
    line.set_series_opts(
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        title_opts=opts.TitleOpts(title="热门城市二手房房价数据", subtitle=f"{data2.index[0]} -- {data2.index[-1]}"),
        toolbox_opts=opts.ToolboxOpts(
            is_show=True, orient="vertical", pos_left="90%",
            feature=opts.ToolBoxFeatureOpts(
                brush=opts.ToolBoxFeatureBrushOpts(type_='clear'),
                data_zoom=opts.ToolBoxFeatureDataZoomOpts(is_show=False),
            ),
        ),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=[opts.DataZoomOpts(is_realtime=True, start_value=60, end_value=100,)],
    )
    components.html(line.render_embed(), width=1200, height=600)

st.write('## :dizzy:相关性分析')
with st.expander('点击查看相关系数'):
    choice = st.selectbox('选择相关系数：', ['Pearson相关系数', 'Kendall相关系数', 'Spearman相关系数', '协方差'])
    if choice == '协方差':
        corr = data2.cov()
    else:
        ind = {'Pearson相关系数': 'pearson', 'Kendall相关系数': 'kendall', 'Spearman相关系数': 'spearman'}
        corr = data2.corr(method=ind[choice])
    fig = ff.create_annotated_heatmap(z=corr.values, x=list(corr.index), y=list(corr.columns),
                                      annotation_text=corr.round(2).values)
    st.plotly_chart(fig, use_container_width=True)


st.write('# :star:广州二手房价数据采集与分析')
st.write('## :dizzy:广州二手房价数据采集')
# ----------任务控制----------
if st.button('开始爬取广州二手房价'):
    # 爬取数据
    gz_data = get_gz_house()
    with st.form('example form4') as f2:  # form表单
        ag4 = AgGrid(
            gz_data,
            height=400,
            fit_columns_on_grid_load=True,  # 列过少的时候，设置True。 列过多的时候就不用设置了
            reload_data=False
        )
        st.form_submit_button()  # 在这里点击提交之后，单元格里面的修改部分就可以传到后面了

gz_data = get_gz_house()
gz_data2 = process_gz(gz_data)

st.write('## :dizzy:广州二手房价词云图')
with st.expander('点击查看'):
    freq = get_freq(gz_data2)
    from pyecharts.charts import WordCloud
    wc = WordCloud()
    wc.add('', ((i, int(j)) for i, j in zip(freq.index, freq.values)), shape=SymbolType.DIAMOND,
           textstyle_opts=opts.TextStyleOpts(font_family="cursive"),)
    components.html(wc.render_embed(), width=1200, height=600)

st.write('## :dizzy:数据分布情况可视化')
with st.expander('点击查看'):
    col1, col2 = st.columns(2)
    with col1:
        choice = st.selectbox('1. 选择查看数据分布情况的列：', ['房屋总价', '房屋单价', '房屋套内面积', '楼龄'])
        chart = st.selectbox('1. 选择查看数据分布的图像类型：', ['直方图', '箱线图'])
        st.subheader(f'{choice}的{chart}可视化')
        if chart == '直方图':
            fig= px.histogram(gz_data2[choice])
            st.plotly_chart(fig, use_container_width=True)
        elif chart == '箱线图':
            fig= px.box(gz_data2[choice])
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        choice = st.selectbox('2. 选择查看数据分布情况的列：', ['房屋总价', '房屋单价', '房屋套内面积', '楼龄'])
        chart = st.selectbox('2. 选择查看数据分布的图像类型：', ['直方图', '箱线图'])
        st.subheader(f'{choice}的{chart}可视化')
        if chart == '直方图':
            fig= px.histogram(gz_data2[choice])
            st.plotly_chart(fig, use_container_width=True)
        elif chart == '箱线图':
            fig= px.box(gz_data2[choice])
            st.plotly_chart(fig, use_container_width=True)

st.write('## :dizzy:广州厅数量、房间数量分布情况')
col1, col2 = st.columns(2)
with col1:
    with st.expander('点击查看厅数量分布情况'):
        st.subheader('厅数量分布情况')
        st.bar_chart(gz_data2['厅数量'].value_counts())

with col2:
    with st.expander('点击查看房间数量分布情况'):
        st.subheader('房间数量分布情况')
        st.bar_chart(gz_data2['房间数量'].value_counts())

st.write('## :dizzy:房屋朝向、装修情况、建筑类型分布情况')
with st.expander('点击查看'):
    choice = st.selectbox('选择需要分析的列', ['房屋朝向', '装修情况', '建筑类型'])
    st.subheader(f'{choice}分布情况')
    fig = px.histogram(data_frame=gz_data2, x=choice, )
    st.plotly_chart(fig, use_container_width=True)

st.write('## :dizzy:广州二手房价与房龄、面积的关系可视化')
col1, col2 = st.columns(2)
with col1:
    with st.expander('点击查看楼龄与总价的关系'):
        st.subheader('楼龄与总价的关系散点图')
        fig = px.scatter(x='楼龄', y='房屋总价', data_frame=gz_data2)
        st.plotly_chart(fig, use_container_width=True)
with col2:
    with st.expander('点击查看房屋面积和价格的关系'):
        st.subheader('房屋套内面积与总价的关系散点图')
        fig = px.scatter(x='房屋套内面积', y='房屋总价', data_frame=gz_data2)
        st.plotly_chart(fig, use_container_width=True)

st.write('## :dizzy:广州二手房价与厅数量、房间数量的关系可视化')
col1, col2 = st.columns(2)
with col1:
    with st.expander('点击查看厅数量与总价的关系'):
        st.subheader('厅数量与总价的关系')
        fig = px.box(pd.pivot(gz_data2, columns='厅数量', values='房屋总价'))
        st.plotly_chart(fig, use_container_width=True)
with col2:
    with st.expander('点击查看房间数量和价格的关系'):
        st.subheader('房间数量和价格的关系')
        fig = px.box(pd.pivot(gz_data2, columns='房间数量', values='房屋总价'))
        st.plotly_chart(fig, use_container_width=True)

st.write('## :dizzy:广州二手房总价与房屋面积、房间数量、厅数量、楼龄的关系')
with st.expander('点击查看相关关系'):
    x = gz_data2[['房屋套内面积', '厅数量', '房间数量', '楼龄']]
    y = gz_data2['房屋总价']
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x, y)
    st.markdown(f'$$y = {model.coef_[0]:.2f}*x_1 + {model.coef_[1]:.2f}*x_2 + {model.coef_[2]:.2f}*x_3 + {model.coef_[3]:.2f}*x_4 + {model.intercept_:.2f}$$')
    st.markdown(
        '''
        - $x_1$: 房屋套内面积
        - $x_2$: 厅数量
        - $x_3$: 房间数量
        - $x_4$: 楼龄
        '''
    )
