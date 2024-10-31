# 问答对RAG应用
### 在线应用
[应用地址](https://app-rag-app-en2r7rktytnwr9rbtappgs.streamlit.app/)（可能会无法使用， 需要点击激活）

### 本地部署
1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. run bge-m3 model service
   
   （注意修改自己的配置）
   
   .streamlit 
   ```
   cp secrets.toml.example secrets.toml
   # 内置token 参见 https://github.com/popjane/free_chatgpt_api
   ```
   run 模型服务
   ```
   $ pip install -r model_requirements.txt
   $ python run_correct_service.py
   ```

3. Run the app
   ```
   $ streamlit run streamlit_app.py
   ```

### 注意
以上demo并不能直接应用到生产环境， 只是作为实现的参考。
