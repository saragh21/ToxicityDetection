# ToxicityDetection
SNLP peroject





---3.12---

Pi Xiaojie: I have crafted a trainable script using bert model (not very sure correct or not, but could be something to start)

- how I train it (I use Aalto Ubuntu VM and vs code)
1. create a conda env using requirements.txt  (there might be some unnecessary library but its fine)
2. connect to aalto shell server
```
ssh USERNAME@kosh.aalto.fi
ssh bach  
```
(different option from (https://www.aalto.fi/en/services/linux-computer-names-in-it-classrooms))
```
module load anaconda
conda init
exec bash
conda activate ... (your env name)
python train.py
```


### some tutorials I followed 

https://github.com/mohammad8921/ToxicCommentDetection-FineTuningBert/blob/main/Codes/BERT_2CNN_1D(max_len_64).ipynb


https://dev.to/alvbarros/toxicity-in-tweets-using-a-bert-model-37in
https://github.com/AlvBarros/toldbr-bert-text-classification-pt-br

https://www.kaggle.com/code/akkefa/finetuning-distilbert-toxic-comment-classification


a tutorial using transformer Trainer to train:
https://medium.com/@heyamit10/fine-tuning-bert-for-classification-a-practical-guide-b8c1c56f252c


there are some long text in the training set. But BERT model can process texts of the maximal length of 512 tokens. but 99% of the training texts are less then 256. (just some findings)