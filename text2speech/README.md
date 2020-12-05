### how to setup??
1. create venv by python 3.7
2. activate venv
3. execute *pip install -r requirements.txt*
4. create a file with the name *venv/lib/python3.7/site-packages/import.pth*. 
   And enter the following in it.
   ```
   {this project path}/epsnet
   {this project path}/ParallelWaveGAN
   ``` 
   
   example
   ```
   /Users/**/PycharmProjects/text2speech-minimum-toolkit/epsnet
   /Users/**/PycharmProjects/text2speech-minimum-toolkit/ParallelWaveGAN
   ``` 
5. Download model_config_file from *https://drive.google.com/drive/folders/1pUm95esEgQGh5HjeLzxpFrZIre7Kqwwd?usp=sharing*.
   And put them into model_config

