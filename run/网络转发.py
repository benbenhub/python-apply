from flask import Flask, request
import requests

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST'])
@app.route('/<path:path>', methods=['GET', 'POST'])
def forward_request(path):
    # 设置转发目标的URL
    forward_url = 'http://{}:8081/{}'.format('${ip}', path)

    # 获取客户端请求信息
    method = request.method
    headers = request.headers
    data = request.get_data()

    # 发送转发请求
    response = requests.request(
        method=method,
        url=forward_url,
        headers=headers,
        data=data,
        stream=True
    )

    # 返回转发响应给客户端
    return response.content, response.status_code, response.headers.items()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)