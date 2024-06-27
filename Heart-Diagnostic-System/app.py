import streamlit as st
from torch import nn
import torch
import librosa
from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import pandas as pd
from models.model.transformer import Transformer
from models.model.con_model import CNN

# 定义组合模型，将CNN和Transformer组合
class CNNTransformerModel(nn.Module):
    def __init__(self, transformer):
        super(CNNTransformerModel, self).__init__()
        self.cnn = CNN()
        self.transformer = transformer
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # 调整形状以匹配Transformer的输入
        x = self.transformer(x)
        return x

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def boundaries_function(denoised_signal):

    # 计算香农能量
    energy = np.power(denoised_signal, 2)
    shannon_energy = -energy * np.log(energy)

    # 计算平均香农能量
    window_size = int(0.02 * 1000)
    hop_length = int(0.01 * 1000)
    shannon_energy_avg = librosa.util.frame(shannon_energy, frame_length=window_size, hop_length=hop_length).mean(axis=0)

    # 标准化香农能量
    shannon_energy_normalized = (shannon_energy_avg - np.mean(shannon_energy_avg)) / np.max(np.abs(shannon_energy_avg))

    # 找到峰值
    peaks, _ = find_peaks(shannon_energy_normalized, height=0.1, distance=window_size//1.5)
    
    num_peaks = len(peaks)
    # 使用过零率算法找到心音边界
    zero_crossings = np.where(np.diff(np.sign(shannon_energy_normalized)))[0]

    # 在每个峰值附近找到最近的过零点
    boundaries = []
    for peak in peaks:
        # 寻找左侧最近的过零点
        left_zero_crossings = zero_crossings[zero_crossings <= peak]
        if len(left_zero_crossings) > 0:
            closest_left_zero_crossing = left_zero_crossings[np.abs(left_zero_crossings - peak).argmin()]
        else:
            closest_left_zero_crossing = None

        # 寻找右侧最近的过零点
        right_zero_crossings = zero_crossings[zero_crossings >= peak]
        if len(right_zero_crossings) > 0:
            closest_right_zero_crossing = right_zero_crossings[np.abs(right_zero_crossings - peak).argmin()]
        else:
            closest_right_zero_crossing = None

        if closest_left_zero_crossing is not None:
            boundaries.append(closest_left_zero_crossing)
        if closest_right_zero_crossing is not None:
            boundaries.append(closest_right_zero_crossing)
    
    return boundaries, num_peaks

def extract_signal(signal, boundaries):

    if len(boundaries) < 22:
        # 如果边界数量不足，返回空结果
        return np.array([]), signal, []
    
    if np.abs(boundaries[4]-boundaries[3]) > np.abs(boundaries[2]-boundaries[1]):
        start_point = boundaries[0]
        end_point = boundaries[19]
    else:
        start_point = boundaries[2]
        end_point = boundaries[21]    
    
    extract_signal1 = signal[start_point*10:end_point*10]
    return extract_signal1

# 初始化session state
if 'denoise_y' not in st.session_state:
    st.session_state.denoise_y = None
if 'segment_y' not in st.session_state:
    st.session_state.segment_y = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'classification_result' not in st.session_state:
    st.session_state.classification_result = ""

st.title('Heart Diagnosis')
uploaded_file = st.file_uploader("upload your audio file", type=['wav'])
st.write('(Minimum 10 seconds)')
if uploaded_file is not None:
    y, sr = librosa.load(uploaded_file, sr=1000)
    st.session_state.uploaded_file = uploaded_file
    st.line_chart(data=y)
row = st.columns(3)
row1 = st.columns(2)

# 创建占位符
denoise_placeholder = st.empty()
segmentation_placeholder = st.empty()
classification_text_placeholder = st.empty()

with row[0]:
    button1 = st.button('denoise')
    if button1:
        if st.session_state.uploaded_file is not None:
            st.session_state.denoise_y = bandpass_filter(y, 20, 400, 1000, order=3)
            denoise_placeholder.line_chart(data=st.session_state.denoise_y)
        else:
            st.write('Please load data first.')
with row[1]:
    button2 = st.button('segmentation')
    if button2:
        if st.session_state.denoise_y is not None:
            boundaries, _ = boundaries_function(st.session_state.denoise_y)
            st.session_state.segment_y = extract_signal(st.session_state.denoise_y, boundaries)
            segmentation_placeholder.line_chart(data=st.session_state.segment_y)
        else:
            st.write("Please perform denoising first.")

with row[2]:
    button3 = st.button('classification')
    if button3:
        if st.session_state.segment_y is not None:
            if len(st.session_state.segment_y) > 5000:
                st.session_state.segment_y = y[:5000]
            else:
                st.session_state.segment_y = np.pad(st.session_state.segment_y, (0, 5000 - len(st.session_state.segment_y)), 'constant')
        
            st.session_state.segment_y = st.session_state.segment_y/np.max(np.abs(st.session_state.segment_y))

            device = torch.device("cpu")
            sequence_len=625 # sequence length of time series
            max_len=5000 # max time series sequence length 
            n_head = 8 # number of attention head
            n_layer = 4# number of encoder layer
            drop_prob = 0.1
            d_model = 64 # number of dimension ( for positional embedding)
            ffn_hidden = 128 # size of hidden layer before classification 
            transformer =  Transformer(  d_model=d_model, n_head=n_head, max_len=max_len, seq_len=sequence_len, ffn_hidden=ffn_hidden, n_layers=n_layer, drop_prob=drop_prob, details=False,device=device).to(device=device)

            model = CNNTransformerModel(transformer).to(device=device)

            model.load_state_dict(torch.load('myModel')) 
            model.eval()
            input_tensor = torch.tensor(st.session_state.segment_y, dtype=torch.float32).to(device)
            input_tensor = input_tensor.unsqueeze(0).unsqueeze(2)  # 变为 [1, 5000, 1]
            assert input_tensor.shape == (1, 5000, 1), f"Expected input shape [1, 5000, 1], but got {input_tensor.shape}"
        
            with torch.no_grad():
                predictions = model(input_tensor)
        
            softmax = nn.Softmax(dim=1)
            probabilities = softmax(predictions)
            _,predicted_classes = torch.max(probabilities, dim=1)
            if predicted_classes == 0:
                st.session_state.classification_result = 'abnormal'
            else:
                st.session_state.classification_result = 'normal'
            classification_text_placeholder.text_area("Classification Result", st.session_state.classification_result, height=20)
        else:
            st.write("Please perform segmentation first.")
