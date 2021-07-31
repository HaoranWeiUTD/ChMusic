# ChMusic Baseline
Baseline solution for Chinese traditional instrument recognition on ChMusic dataset
KNN_ChMusic.py use MFCCs as feature and KNN as classification model.
CNN_ChMusic.py use MFCCs as feature and Convolution neural network as classification model.

# ChMusic Dataset
ChMusic is a traditional Chinese music dataset for training model and performance evaluation of musical instrument recognition. 
This dataset cover 11 musical instruments, consisting of Erhu, Pipa, Sanxian, Dizi, Suona, Zhuiqin, Zhongruan, Liuqin, Guzheng, Yangqin and Sheng.

Each musical instrument has 5 traditional Chinese music excerpts, so there are 55 traditional Chinese music excerpts in this dataset. The name of each music excerpt and the corresponding musical instrument is shown on Table~\ref{tab:Musics}. Each music excerpt only played by one musical instrument in this dataset.

Each music excerpt is saved as a .wav file. The name of these files follows the format of "x.y.wav", "x" indicates the instrument number, ranging from 1 to 11, and "y" indicates the music number for each instrument, ranging from 1 to 5. These files are recorded by dual channel, and has sampling rate of 44100 Hz. The duration of these music excerpts are between 25 to 280 seconds.

This ChMusic dataset taks 530M and can be download from Baidu Wangpan by link 
pan.baidu.com/s/13e-6GnVJmC3tcwJtxed3-g and password xk23, 
or from google drive 
drive.google.com/file/d/1rfbXpkYEUGw5h_CZJtC7eayYemeFMzij/view?usp=sharing

# Acknowledgment
Thanks for Zhongqin Yu, Ziyi Wei, Baojin Qin, Guangyu Zhao, Qiuxiang Wang, Xiaoqian Zhang, Jiali Yao, Zheng Gao, Ke Yan, Menghao Cui and Yichuan Zhang for playing musics and helping to collect this ChMusic dataset.
