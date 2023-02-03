#Thinkbig_KoGPT2_fine_tunning
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import re
from tqdm import tqdm

# 스페셜 토큰
U_TKN = '<usr>' #Qusetion토큰
S_TKN = '<sys>' #Answer토큰
BOS = '</s>'#문장의 시작 토큰
EOS = '</s>'#문장의 끝 토큰
MASK = '<unused0>'#마스크 토큰
SENT = '<unused1>'#문장 토큰(Q와 A토큰 사이에 넣어서 구분)
PAD = '<pad>' #패드 토큰

# #hugging_face의 KoGPT2(이미 학습된 데이터)를 가져옴
koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK)
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

# 파라미터, 크로스엔트로피로스, 옵티마이저(아담)
epoch = 2
Sneg = -1e18
learning_rate = 3e-5
criterion = torch.nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 전처리된 데이터 불러오기
df = pd.read_csv('../../ChatbotData.csv')
df.head()

#데이터셋 클래스 상속
class ChatbotDataset(Dataset):
    # 데이터셋의 전처리를 해주는 부분
    def __init__(self, chats, max_len=64):  
        self._data = chats
        self.max_len = max_len
        self.q_token = U_TKN
        self.a_token = S_TKN
        self.sent_token = SENT
        self.eos = EOS
        self.pad = PAD
        self.mask = MASK
        self.tokenizer = koGPT2_TOKENIZER

    def __len__(self):
        return len(self._data)

    #Q,A만 사용하여 파인튜닝을 위한 토큰화(인덱스(idx)에 해당하는 입출력 데이터 반환)
    def __getitem__(self, idx):  
        turn = self._data.iloc[idx]
        q = turn['Q']  # 질문을 가져온다.
        q = re.sub(r'([?.!,])', r' ', q)  # 특수기호 생략(이거 안하면 결과가 이상하게 나올 때가 많음)

        a = turn['A']  # 답변을 가져온다.
        a = re.sub(r'([?.!,])', r' ', a)  

        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)
        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        #질문의 길이가 최대길이(64)보다 크면 
        if q_len > self.max_len:
            a_len = self.max_len - q_len                        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:                                      #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len                    #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        #질문의 길이 + 답변의 길이가 최대길이보다 크면
        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len                        #답변의 길이를 최대길이 - 질문길이
            if a_len <= 0:                                      #질문의 길이가 너무 길어 질문만으로 최대 길이를 초과 한다면
                q_toked = q_toked[-(int(self.max_len / 2)) :]   #질문길이를 최대길이의 반으로 
                q_len = len(q_toked)
                a_len = self.max_len - q_len                    #답변의 길이를 최대길이 - 질문길이
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        # 답변 labels = [mask, mask, ...., mask, ..., <bos>,..답변.. <eos>, <pad>....]
        labels = [self.mask] * q_len + a_toked[1:]

        # mask = 질문길이 0 + 답변길이 1 + 나머지 0
        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # 답변 labels을 index 로 만든다.
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        # 최대길이만큼 PADDING
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        # 질문 + 답변을 index 로 만든다.    
        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        # 최대길이만큼 PADDING
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        #질문+답변, 마스크, 답변
        return (token_ids, np.array(mask), labels_ids)

# batches가 1이 아닌 경우 이런식으로 세팅하여 DataLoader의 collate_fn에 넣어준다.
def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)

df=df[['Q','A']]
# df = df.iloc[:100,:] #테스트 시 데이터를 짧게 만들어서 구동여부 확인

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'GPU 사용 가능한가요 ? : {torch.cuda.is_available()}') 


train_set = ChatbotDataset(df, max_len=64) 
#윈도우 환경에서 num_workers 는 무조건 0으로 지정
train_dataloader = DataLoader(train_set, batch_size=32, num_workers=0, shuffle=True, collate_fn=collate_batch)
model.to(device)
model.train()

print ('학습 시작')
for epoch in range(epoch):
    for batch_idx, samples in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad() #Pytorch에서는 gradients값들을 추후에 backward를 해줄때 계속 더해주기 때문
        token_ids, mask, label = samples
        out = model(token_ids)
        out = out.logits      #Returns a new tensor with the logit of the elements of input
        mask_3d = mask.unsqueeze(dim=2).repeat_interleave(repeats=out.shape[2], dim=2)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out))
        loss = criterion(mask_out.transpose(2, 1), label)
        # 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        # 학습 끝
        optimizer.step()#경사하강법(gradient descent)
print ('학습 종료')

### 챗봇 실행 'quit' 입력 시 종료
with torch.no_grad(): #requires_grad=False 상태가 되어 메모리 사용량 아껴줌
    print('챗봇 작동 중입니다. 종료를 원하면 \"quit\"을 입력해주세요')
    print(' ')
    while True :
        q = input('나 > ').strip()
        if q == 'quit':
            break
        a = ''
        while True:
            input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(U_TKN + q + SENT + S_TKN + a)).unsqueeze(dim=0)
            pred = model(input_ids)
            pred = pred.logits
            #마지막 dim의 최대값 인덱스 
            gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().numpy().tolist())[-1]
            if gen == EOS:
                break
            a += gen.replace('▁', ' ')
        print('Chatbot > {}'.format(a.strip()))
