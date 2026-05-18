# -*- coding: utf-8 -*-
"""
app.py
纯 NumPy 驱动的滨海高桩承台桥墩地震失效模式预测系统 (Web 网页版)
"""

import streamlit as st
import numpy as np
import joblib
import os

# ================== 1. 网页全局配置 ==================
st.set_page_config(
    page_title="Seismic Failure Mode Test",
    layout="wide", # 使用宽屏模式以容纳左右分栏
    initial_sidebar_state="collapsed"
)

# 注入自定义 CSS，压缩表单行距，使其像 PyQt5 界面一样紧凑严谨
st.markdown("""
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1400px; }
    div[data-testid="column"] { display: flex; flex-direction: column; justify-content: center; }
    hr { margin-top: 10px; margin-bottom: 20px; }
    
    /* 新增：让所有输入框内的数字居中对齐 */
    div[data-baseweb="input"] input {
        text-align: center !important;
    }  
    </style>
""", unsafe_allow_html=True)


# ================== 2. 模型加载缓存 ==================
@st.cache_resource
def load_model():
    assets_path = 'model_assets_numpy.pkl'
    if not os.path.exists(assets_path):
        return None
    return joblib.load(assets_path)

assets = load_model()


# ================== 3. 核心界面布局 ==================
# 按照你 PyQt5 的比例，左侧输入占约 60%，右侧输出占约 40%
col_left, spacer, col_right = st.columns([5.5, 0.5, 4])

# ----------------- 左侧：19个参数 4列布局区 -----------------
with col_left:
    st.markdown("<h4 style='color: #333;'>Input: Bridge design parameters</h4>", unsafe_allow_html=True)
    
    # 表头设计
    h1, h2, h3, h4 = st.columns([1.5, 3.5, 1.5, 1.5])
    h1.markdown("<div style='text-align: center; color: #800020; font-weight: bold;'>Parameter</div>", unsafe_allow_html=True)
    h2.markdown("<div style='text-align: center; color: #800020; font-weight: bold;'>Description</div>", unsafe_allow_html=True)
    h3.markdown("<div style='text-align: center; color: #800020; font-weight: bold;'>Range</div>", unsafe_allow_html=True)
    h4.markdown("<div style='text-align: center; color: #800020; font-weight: bold;'>Value</div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 5px 0 15px 0;'>", unsafe_allow_html=True)

    # 19 个参数配置表 (标识符, HTML格式, 描述, 范围, 最小值, 最大值, 默认值, 步长, 格式)
    params_config = [
        ("N", "N", "Number of pile rows along the loading direction", "2~4", 2.0, 4.0, 3.0, 1.0, "%.0f"),
        ("Dp", "D<sub>p</sub> (m)", "Pile diameter", "0.6~1.8", 0.6, 1.8, 1.2, 0.1, "%.2f"),
        ("rho_pl", "ρ<sub>pile,l</sub>", "Pile longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, 0.001, "%.3f"),
        ("alpha", "α", "Column axial load ratio", "0.05~0.25", 0.05, 0.25, 0.15, 0.01, "%.2f"),
        ("S_Dp", "S (D<sub>p</sub>)", "Pile spacing", "2.5~3.5", 2.5, 3.5, 3.0, 0.1, "%.2f"),
        ("Dr", "D<sub>r</sub>", "Sand density", "0.35~0.75", 0.35, 0.75, 0.55, 0.05, "%.2f"),
        ("SD", "SD (m)", "Scour depth", "0~6", 0.0, 6.0, 3.0, 0.5, "%.2f"),
        ("Hp_Dc", "H<sub>p</sub>/D<sub>c</sub>", "Column aspect ratio", "0~4", 0.0, 4.0, 2.0, 0.1, "%.2f"),
        ("Dc_Dp", "D<sub>c</sub> (D<sub>p</sub>)", "Column diameter", "1.5~2.5", 1.5, 2.5, 2.0, 0.1, "%.2f"),
        ("rho_cl", "ρ<sub>column,l</sub>", "Column longitudinal reinforcement ratio", "0.005~0.015", 0.005, 0.015, 0.010, 0.001, "%.3f"),
        ("rho_ps", "ρ<sub>pile,s</sub>", "Pile transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, 0.001, "%.3f"),
        ("fyl", "f<sub>yl</sub> (MPa)", "Longitudinal rebar yield strength", "300~500", 300.0, 500.0, 400.0, 10.0, "%.0f"),
        ("fc", "f<sub>c</sub> (MPa)", "Concrete compressive strength", "20~50", 20.0, 50.0, 35.0, 1.0, "%.1f"),
        ("rho_cs", "ρ<sub>column,s</sub>", "Column transverse reinforcement ratio", "0.003~0.013", 0.003, 0.013, 0.008, 0.001, "%.3f"),
        ("Xt", "X<sub>t</sub> (X<sub>l</sub>)", "Corrosion level of transverse reinforcement", "1~3", 1.0, 3.0, 2.0, 0.1, "%.2f"),
        ("Xl", "X<sub>l</sub>", "Corrosion level of longitudinal reinforcement", "0~0.30", 0.0, 0.30, 0.15, 0.01, "%.2f"),
        ("t", "t (m)", "Column cover concerete thickeness", "0.04~0.08", 0.04, 0.08, 0.06, 0.01, "%.2f"),
        ("d_l", "d<sub>l</sub> (m)", "Column longitudinal reinforcement diameter", "0.018~0.032", 0.018, 0.032, 0.025, 0.001, "%.3f"),
        ("fyt", "f<sub>yt</sub> (MPa)", "Transverse rebar yield strength", "200~400", 200.0, 400.0, 300.0, 10.0, "%.0f")
    ]

    raw_features = []
    
    # 动态渲染 19 个参数行
    for p_id, html_name, desc, rng, p_min, p_max, p_default, p_step, p_format in params_config:
        c1, c2, c3, c4 = st.columns([1.5, 3.5, 1.5, 1.5])
        
        # 垂直居中渲染文本
        c1.markdown(f"<div style='text-align: center; color: #4a235a; font-weight: bold; font-family: Arial; padding-top: 8px;'>{html_name}</div>", unsafe_allow_html=True)
        c2.markdown(f"<div style='text-align: center; color: #444444; font-size: 14px; padding-top: 8px;'>{desc}</div>", unsafe_allow_html=True)
        c3.markdown(f"<div style='text-align: center; color: #666666; font-size: 14px; padding-top: 8px;'>{rng}</div>", unsafe_allow_html=True)
        
        # 输入框
        with c4:
            val = st.number_input(
                label=p_id,
                min_value=p_min, max_value=p_max, value=p_default, step=p_step, format=p_format,
                label_visibility="collapsed",
                key=p_id
            )
            raw_features.append(val)

# ----------------- 右侧：控制与输出区 -----------------
with col_right:
    # 作者信息 (通过 margin-top 往下微调位置)
    st.markdown("""
        <div style='text-align: right; color: #555555; line-height: 1.5; font-family: Arial; font-size: 14px; margin-top: 25px;'>
            Created by Jingcheng Wang, Associate Professor.<br>Fuzhou University
        </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    
    # Predict 按钮
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)
    st.write("")

    st.markdown("<h5 style='color: #333; font-family: Arial; font-weight: bold;'>Failure Mode Prediction</h5>", unsafe_allow_html=True)

    # 渲染带有百分比的自定义进度条的 HTML 函数
    def draw_progress_bar(label, percentage, color_hex):
        # 智能颜色判断：进度大于 50% 用白色，小于 50% 用深灰色
        text_color = "white" if percentage > 50 else "#333333"
        
        html = f"""
        <div style="margin-bottom: 15px;">
            <div style="font-weight: bold; font-family: Arial; font-size: 14px; margin-bottom: 5px;">{label}</div>
            <div style="width: 100%; background-color: #f0f0f0; border-radius: 4px; border: 1px solid #ccc; position: relative; height: 28px;">
                <div style="width: {percentage}%; background-color: {color_hex}; height: 100%; border-radius: 3px; transition: width 0.6s ease-in-out;"></div>
                <div style="position: absolute; width: 100%; text-align: center; top: 0; left: 0; line-height: 28px; font-weight: bold; font-family: Arial; font-size: 13px; color: {text_color};">{percentage:.1f}%</div>
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # 模型推理逻辑
    probs = [0.0, 0.0, 0.0] # 默认概率
    
    if predict_clicked:
        if assets is None:
            st.error("⚠️ 未检测到 model_assets_numpy.pkl，请将模型权重文件与代码放在同一目录下。")
        else:
            weights = assets['weights']
            scaler = assets['scaler']
            label_names = assets['le'].classes_
            
            # NumPy 前向传播
            input_vector = np.array(raw_features).reshape(1, -1)
            a = scaler.transform(input_vector)
            
            for w, b in weights[:-1]:
                z = np.dot(a, w) + b
                a = np.maximum(0, z) # ReLU
                
            w_out, b_out = weights[-1]
            z_out = np.dot(a, w_out) + b_out
            
            exp_z = np.exp(z_out - np.max(z_out, axis=1, keepdims=True)) 
            probs_array = (exp_z / np.sum(exp_z, axis=1, keepdims=True))[0]
            
            # 对齐概率与标签 ['FFF', 'PFF', 'PSF']
            probs_dict = {label_names[idx]: float(probs_array[idx]) for idx in range(len(label_names))}
            
            # 按 PFF, FFF, PSF 顺序取出
            probs = [probs_dict.get('PFF', 0.0), probs_dict.get('FFF', 0.0), probs_dict.get('PSF', 0.0)]

    # 绘制三大进度条
    draw_progress_bar("PFF (Pier Flexure Failure)", probs[0] * 100, "#0078d4") # 蓝
    draw_progress_bar("FFF (Foundation Flexure Failure)", probs[1] * 100, "#008000") # 绿
    draw_progress_bar("PSF (Pier Shear Failure)", probs[2] * 100, "#d83b01") # 红

    st.write("---")
    
    # 结构示意图加载
    try:
        st.image("structure.png", use_container_width=True)
    except:
        st.markdown("""
        <div style='border: 1px dashed #ccc; padding: 40px; text-align: center; color: #999; font-family: Arial;'>
            Structure Image<br>(structure.png not found)
        </div>
        """, unsafe_allow_html=True)