import pandas as pd
import streamlit as st
from pyecharts.charts import Sankey
from pyecharts import options as opts
from streamlit_echarts import st_pyecharts

# 提取路径信息（优化版）
def extract_paths_optimized(df):
    # 按码和单据时间进行排序
    df_sorted = df.sort_values(by=['码', '单据时间ymd'])

    # 使用 groupby 提取每个码的路径信息
    def get_path(group):
        path = ' -> '.join(group['发货企业'].tolist() + [group['收货企业'].iloc[-1]])
        return path

    # 针对每个唯一的码，提取路径
    paths = df_sorted.groupby('码', group_keys=False).apply(get_path).reset_index()
    paths.columns = ['码', '路径']
    
    return paths

# 创建Sankey图
def create_sankey(df, batch_number, store, month, sku, brand, subbrand, doc_type, threshold, percent_threshold):
    # 填补缺失值
    df.fillna('缺失值', inplace=True)
    
    # 过滤
    filtered_df = df[
        ((df['批次号'] == batch_number) | (batch_number == '全选')) & 
        ((df['店铺'] == store) | (store == '全选')) &
        ((df['月份'] == month) | (month == '全选')) &
        ((df['sku'] == sku) | (sku == '全选')) &
        ((df['brand'] == brand) | (brand == '全选')) &
        ((df['subbrand'] == subbrand) | (subbrand == '全选')) &
        ((df['单据类型'] == doc_type) | (doc_type == '全选'))
    ]

    # 统计批次号下的总码数量（去重后）
    total_codes = filtered_df['码'].nunique()

    # 去重, 统计发货企业到收货企业的码数量
    df_grouped = filtered_df.groupby(['发货企业', '收货企业'])['码'].nunique().reset_index()
    df_grouped.columns = ['source', 'target', 'value']

    # 计算百分比
    df_grouped['percentage'] = df_grouped['value'] / total_codes * 100

    # 根据阈值更新target
    df_grouped['target'] = df_grouped.apply(
        lambda row: "Others" if row['value'] < threshold or row['value'] < total_codes * percent_threshold else row['target'], axis=1
    )

    nodes = list({v for v in df_grouped["source"]}.union({v for v in df_grouped["target"]}))
    nodes = [{"name": node} for node in nodes]

    links = [
        {"source": row['source'], "target": row['target'], "value": row['value'], "percentage": row['percentage']}
        for _, row in df_grouped.iterrows()
    ]

    # 使用优化后的路径提取方法
    flow_data = extract_paths_optimized(filtered_df)

    c = (
        Sankey(init_opts=opts.InitOpts(width="1600px", height="800px"))
        .add(
            "sankey",
            nodes,
            links,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
            tooltip_opts=opts.TooltipOpts(
                formatter=lambda params: f"{params['data']['source']} → {params['data']['target']}<br>码数量: {params['data']['value']}<br>占比: {params['data']['percentage']:.2f}%"
            ),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Sankey-追溯码"))
    )
    
    return c, flow_data, total_codes, filtered_df

# Streamlit 应用
st.set_page_config(layout="wide")  # 设置画幅变宽
st.title("Sankey 图 - 追溯码分析")

# 上传文件单元框
uploaded_file = st.file_uploader("上传一个文件", type=["xlsx", "csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)

    # 获取唯一值，添加“全选”，排序
    def get_unique_sorted_values(column):
        unique_values = df[column].fillna('缺失值').unique().tolist()
        return ['全选'] + sorted(unique_values)

    # 初始化筛选条件的选项
    batch_numbers = get_unique_sorted_values('批次号')
    stores = get_unique_sorted_values('店铺')
    months = get_unique_sorted_values('月份')
    skus = get_unique_sorted_values('sku')
    brands = get_unique_sorted_values('brand')
    subbrands = get_unique_sorted_values('subbrand')
    doc_types = get_unique_sorted_values('单据类型')

    # 创建下拉筛选器和阈值输入框，每行三个
    col1, col2, col3 = st.columns(3)
    with col1:
        batch_number = st.selectbox('批次号', batch_numbers)
    with col2:
        store = st.selectbox('店铺', stores)
    with col3:
        month = st.selectbox('月份', months)

    col4, col5, col6 = st.columns(3)
    with col4:
        sku = st.selectbox('SKU', skus)
    with col5:
        brand = st.selectbox('品牌', brands)
    with col6:
        subbrand = st.selectbox('子品牌', subbrands)

    col7, col8, col9 = st.columns(3)
    with col7:
        doc_type = st.selectbox('单据类型', doc_types)
    with col8:
        threshold = st.number_input('数量阈值:', min_value=0, value=0)
    with col9:
        percent_threshold = st.number_input('百分比阈值:', min_value=0.0, max_value=1.0, value=0.0)

    # 更新 Sankey 图
    if st.button('更新图表'):
        c, flow_data, total_codes, filtered_df = create_sankey(df, batch_number, store, month, sku, brand, subbrand, doc_type, threshold, percent_threshold)
        
        # 实时显示非重复码的数量
        st.markdown(f"### 此批次号下的码的计数(非重复): {total_codes}")
        
        # 渲染 Sankey 图
        st_pyecharts(c)

        # 只展示筛选后的路径信息
        filtered_paths = flow_data[flow_data['码'].isin(filtered_df['码'])]  # 只提取筛选后的码对应的路径信息
        st.markdown("### 路径信息")
        for _, row in filtered_paths.iterrows():
            st.markdown(f"码: {row['码']} - 路径: {row['路径']}")
