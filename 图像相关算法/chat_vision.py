import os
import cv2
import requests as rq
from ollama import Client

ollama_vlm_model = "gemma3:4b"
ollama_url ="http://127.0.0.1:11434"   #"http://172.20.10.5:11434"
cam_num = 0


def run_chat_vision(question):
    try:
        rq.get(ollama_url)
    except:
        os.system(f"ollama pull {ollama_vlm_model}")
    cap = cv2.VideoCapture(cam_num, cv2.CAP_DSHOW)      #打开摄像头
    if not cap.isOpened():
        return "无法打开摄像头"
    ret, frame = cap.read() #读取一帧图像
    cap.release()  #释放
    _, buffer = cv2.imencode('.jpg', frame)
    byte_data = buffer.tobytes()
    client = Client(host=ollama_url)
    response = client.chat(model=ollama_vlm_model,
                           messages=[{'role': 'user', 'content': question, 'images': [byte_data]}])
    return response['message']['content']


if __name__ == "__main__":
    while True:
        text = input("请输入提问内容(例如你看到了什么)：")
        try:
            answer = run_chat_vision(text)
            print("AI：", answer)
        except:
            print(f"提示: 请确保ollama已经运行{ollama_vlm_model}，再进行聊天。")
