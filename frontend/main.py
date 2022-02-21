from cgitb import text
import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

IMG_SIZE = 50

def change_img(upfile):
    image = Image.open(upfile)
    with io.BytesIO() as output:
        image.save(output, format='JPEG')
        binary_img = output.getvalue()
    # image = image.convert('RGB')
    # image = image.resize((IMG_SIZE, IMG_SIZE))
    # data = np.asarray(image)/255
    return binary_img

st.title("猫ちゃん推論web app")
st.markdown("アビシニアン、ブリティッシュショートヘア、ロシアンブルー, メインクーンに分類できます。")

upfile = st.file_uploader("写真を選択して下さい", type=['png', 'jpg'])
start = st.button("スタート")

if start:
    if upfile is not None:
        data = change_img(upfile)
        files = {"in_file": data}
        res = requests.post(f"http://backend:8080/upload", files=files)
        # st.write("推論を開始します")
        #結果の表示
        st.markdown("***")
        st.markdown("## 推論結果")
        ans = res.json()["text"]
        st.markdown(f"あなたのねこちゃんは…**{ans}**です！")

        image=Image.open(upfile)
        img_array = np.array(image)
        st.image(img_array, use_column_width = True)
