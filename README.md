# accent generator
### Setup
1. Follow accent_generator/text2speech/README.md
2. Setup crepe
    ```
    $ pip install --upgrade tensorflow  # if you don't already have tensorflow >= 2.0.0
    $ pip install crepe
    ```

### How to move
Set the taget wavfile and its duration in the target directory, and write its phonemes inside main.py. 
It also requires the path to the wavfile and durationfile inside main.py.
```
#example
accent_generator/target/sample.wav
accent_generator/target/duration
#example of the durationfile: 4 14 11 2 9 5 7 4 9 6 7 7 0 6 9 5 6 3 12 24 6 7 4 5 5 4 5 4 3 7 5 7 7 8 7 8 4 7 5 5 7 3 3 3 6 4 4 5 3 5 7 2 9 3 4 7 4 4 7 7 3 0 8
```
```
#inside main.py
input_wav = "./target/sample.wav"
input_phonemes = [] 
#example: [7 ,6 ,6 ,10 ,2 ,12, 11 ,6, 6, 11 ,2 ,12 ,8 ,4 ,3, 7 ,6 ,10 ,5, 14 ,11, 2, 10, 5 ,13 ,3 ,8, 6, 10, 2, 3, 11, 6, 12, 3 ,22 ,3, 3 ,15 ,4 ,11 ,5, 10 ,5 ,7, 3, 9, 3 ,17, 2 ,20, 4 ,24, 5 ,21, 3 ,3 ,16, 6, 11, 5 ,40]
duration = "./target/duration"
```
Execute`python main.py`

### アルゴリズムの説明
1. まずdurationをもとに目標データの音素ごとのF0平均をとる
2. F0が極大値となっている音素(つまり直前と直後よりもF0が高い音素)にアクセント核"1"をつける。ただしアクセント核が近い位置に来ることはないので"1 0 1"や"1 0 0 1"となった場合は最初のアクセント核を消す("0 0 1","0 0 0 1"にする)
3. こうしてできたaccentの一つ一つのアクセント核について、前後二つずつを含めた計5つの音素で32通りのアクセントを試し("0 0 0 0 0","0 0 0 0 1",...,"0 1 1 0 1","0 1 1 1 0",...,"1 1 1 1 1")、最も目標データとのF0の距離が小さかったものを選択しaccentを更新
4. これを数回繰り返す(デフォルトでは1回)

※ 3での距離の計算方法
->0.01秒ごとのF0の二乗誤差をとり、平均する
