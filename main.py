import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

st.set_page_config(layout="wide")  # 전체 너비 사용

st.title("<3 in 1> Complementary Coffee Cups")
st.markdown("<h3 style='color:#567ace'><center>3114 최유성</center></h3><p></p>", unsafe_allow_html=True)

# 좌우 컬럼 나누기
left_col, right_col = st.columns([1, 2])  # 왼쪽: 제어 UI, 오른쪽: 그래프

with left_col:
    st.markdown("#### 1. 컵의 높이 h를 선택하세요", unsafe_allow_html=True)
    st.markdown("※ [빨간색] 함수 선택 시 높이는 2보다 작아야 함")
    # 슬라이더로 h값 설정
    h = st.slider("", min_value=1., max_value=12., value=8., step=0.1)
    # h = st.slider("", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    # 함수 선택
    st.markdown("#### 2. 함수 x=f(y)를 선택하세요 (하단 DESMOS 참고)", unsafe_allow_html=True)
    option = ("[파란색]   x = -(1/16)*(y - 4)^2 + 4", "[빨간색]   x = exp(-y^2)")
    func_option = st.selectbox("", option)

    st.markdown(
        '<p></p><iframe src="https://www.desmos.com/calculator/mxwfexo44e?embed" width="400" height="400" style="border: 1px solid #ccc" frameborder=0></iframe>',
        unsafe_allow_html=True)

# 함수 정의
q = sp.symbols('q')
f_expr = sp.exp(-q ** 2)
f2_expr = -(1 / 16) * (q - 4) ** 2 + 4

def f(y):
    f_sympy = sp.lambdify(q, f_expr, modules=['numpy'])  # numpy 호환용 함수
    return f_sympy(y)

def f2(y):
    f2_sympy = sp.lambdify(q, f2_expr, modules=['numpy'])
    return f2_sympy(y)

# 선택한 함수에 따라 expr와 함수 할당
if func_option == option[1]:
    selected_expr = f_expr
    selected_func = f
else:
    selected_expr = f2_expr
    selected_func = f2

# k 계산 및 출력
k = float(2 * (sp.integrate(selected_expr, (q, 0, h))) / h)

with left_col:
    st.markdown(f"#### -> 두 컵에 같은 양의 커피를 담을 수 있는", unsafe_allow_html=True)
    st.markdown(f"#### <u>k = {k:.4f}</u>", unsafe_allow_html=True)
    st.markdown(f"<br></br>※ k는 어떻게 구할까?", unsafe_allow_html=True)
    st.markdown('<a href="https://freeimage.host/i/3vsY4v2"><img src="https://iili.io/3vsY4v2.md.png"></a>', unsafe_allow_html=True)



# g 정의
def g(y):
    return k - selected_func(y)

# y값 설정
y = np.linspace(0, h, 200)
theta = np.linspace(0, 2 * np.pi, 100)
Y, Theta = np.meshgrid(y, theta)

# 컵 A
X_A = selected_func(Y) * np.cos(Theta)
Z_A = selected_func(Y) * np.sin(Theta)
Y_A = Y

# 컵 B
X_B = g(Y) * np.cos(Theta) + k
Z_B = g(Y) * np.sin(Theta)
Y_B = Y

# 그래프
fig = go.Figure()
fig.add_trace(go.Surface(x=X_A, y=Y_A, z=Z_A, colorscale='Blues', opacity=0.7, showscale=False))
fig.add_trace(go.Surface(x=X_B, y=Y_B, z=Z_B, colorscale='Oranges', opacity=0.7, showscale=False))

flag = True
fig.update_layout(
    width=700,
    height=500,
    scene=dict(
        xaxis=dict(visible=flag),
        yaxis=dict(visible=flag),
        zaxis=dict(visible=flag),
        aspectratio=dict(x=1.6, y=1, z=1.2),
        camera=dict(eye=dict(x=0, y=-1.5, z=1)),
    )
)

with right_col:
    st.markdown('3D로 시각화된 Complementary Coffee Cups를 자유자재로 돌려보세요!')
    st.plotly_chart(fig)
