# chinese correction app demo

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

3. run corrector model service

   .streamlit create file secrets.toml
   ```
   [remote]
   algo_url="http://0.0.0.0:9046/app/rag"
   openai_api_key="xxx"
   openai_url="https://free.gpt.ge/v1/"
   ```

   run 
   ```
   $ python run_rag_service.py
   ```


