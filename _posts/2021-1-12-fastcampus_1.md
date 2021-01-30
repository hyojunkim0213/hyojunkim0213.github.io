---
layout: post
title: "PyTorch Lecture about Machine Learning - 9: Softmax Classifier"
date: 2021-01-10
excerpt: "Sung Kim 님의 PyTorch lecture를 공부하고 작성하고자 합니다."
nlp_fastcampus: true
tag:
- PyTorch
comments: false
---

- 지난 강의

1. [PyTorch Lecture about Machine Learning - 3: Gradient Descent]({% post_url 2020-12-28-lecture3 %})
2. [PyTorch Lecture about Machine Learning - 4: Back-propagation and Autograd]({% post_url 2020-12-30-lecture4 %})
3. [PyTorch Lecture about Machine Learning - 5 & 6: Linear Regression in PyTorch way]({% post_url 2021-1-2-lecture5&6 %})
4. [PyTorch Lecture about Machine Learning - 7 & 8: Wide & Deep model + DataLoader]({% post_url 2020-12-30-lecture4 %})

## Softmax

지난 시간에는 (N * 2) Matrix 의 x와 (2 * 1) Matrix 의 w가 곱해져 (N * 1) Matrix의 y를 출력하였다.<br>

MNIST data의 경우 1~10까지 label이 10개이며 output이 10개라 할 수 있다.<br>
이러한 경우, (2 * 10) Matirx의 w를 곱하여 neural network를 생성할 수 있다.

출력층이 10개인 모델의 확률을 계산하는 방법은 **Softmax** 함수를  이용하는 것이다.

<center><img src="https://user-images.githubusercontent.com/28617444/104113408-d57fd580-533c-11eb-90fe-7ad86a8e3807.png" width="400" height="100"></center>
<br>
모든 출력값을 0과 1 사이의 값으로 반환하는 것이다.<br>
이 때, 반환된 값들의 총합은 **1**이 되기 때문에 probability의 개념을 적용할 수 있다.


![image](https://user-images.githubusercontent.com/28617444/104113437-1a0b7100-533d-11eb-9295-fb195e57375c.png)

10개의 probability에 대해 one hot vector의 개념으로 one hot label을 부여해준다. 즉, 확률이 가장 높은 label을 1로 나머지를 0으로 부여하는 것이다. 위의 예제에서는 y=0 인 경우가 1, 나머지는 0이 되겠다.

그 후, one hot label과 실제 값의 차이를 통해 loss를 계산해야 한다. <br>
이 때 손실함수로 **cross entropy** 를 사용한다.

<center><img src="https://user-images.githubusercontent.com/28617444/104113506-b7ff3b80-533d-11eb-9288-c3412dad2fe2.png" width="400" height="100"></center>

## Cross entropy in PyTorch

PyTorch에서 Cross entropy를 계산함에 있어서 두 가지 장점이 있다.
1. Y value는 one hot 이 아닌, class label 숫자 하나만 주면 된다.

2. Softmax 함수 계산 없이, 모델을 통한 pred 값 그대로 넣어준다.<br>
CrossEntropyLoss() 에서 Softmax  계산이 포함되어 있기 때문이다.

```python
loss = nn.CrossEntropyLoss()

Y = Variable(torch.LongTensor([0]), requires_grad = False)

Y_pred1 = Variable(torch.Tensor[[2.0, 1.0, 0.1]])
Y_pred2 = Variable(torch.Tensor[[0.5, 2.0, 0.3]])

l1 = loss(Y_pred1, Y)
l2 = loss(Y_pred2, Y)
```

## Batch mode

다음으로 PyTorch 에서 장점은 배치 모드를 사용할 수 있는 것이다.

multiple label에 대한 multiple prediction 을 한번에 계산할 수 있다.

```python
loss = torch.nn.CrossEntropyLoss()

Y=torch.LongTensor([2,0,1] , requires_grad=False)

Y_pred1 = torch.Tensor([[0.1,0.2,0.9],  
                       [1.1,0.1,0.2],
                       [0.2,2.1,0.1]])
Y_pred2 = torch.Tensor([[0.8,0.2,0.3],
                       [0.2,0.3,0.5],
                       [0.2,0.2,1.5]])

l1 = loss(Y_pred1,Y)
l2 = loss(Y_pred2,Y)

print(l1,l2)
```
pred1의 경우 첫번째는 0.9로 2의 인덱싱의 숫자, <br>
두번째는 1.1로 0의 인덱싱의 숫자, <br>
세번째는 2.1로 1의 인덱싱의 숫자로 모두 예측하였다. <br>

l1의 loss는 0.5로 작고, l2의 losssms 1.24로 큰 편이다. <br>
즉, Y_pred1이 더 좋은 예측임을 확인할 수 있다.


## MNIST input

MNIST 데이터 하나는 28 * 28 의 픽셀로 이루어져 있으며 0~9까지의 손글씨를 나타낸다.

![image](https://user-images.githubusercontent.com/28617444/105053267-18b41400-5ab4-11eb-9195-bcb84d318a1d.png)

그림 상으로 보았을 때, input layer 는 784 픽셀을 의미하며 output layer는 0~9까지의 label이 부여되어 있다.

또한 사이에 존재하는 layer를 **hidden layer** 라고 하며 원하는만큼 노드를 구성할 수 있다.

PyTorch에서는 다음과 같이 구성한다.

self.l1 = nn.Linear(inputsize , outsize)<br>
또한, layer마다 outputsize와 inputsize는 맞물리며 구성되어야 한다.<br>
다음은 하나의 예이다.

```python
self.l1 = nn.Linear(784, 520)
self.l2 = nn.Linear(520,320)
self.l3 = nn.Linear(320,240)
self.l4 = nn.Linear(240,120)
self.l5 = nn.Linear(120,10)
```
마지막 layer는 10개의 숫자를 예측하기 때문에 10의 output layer로 구성되어 있다.

앞에서 정의한
```python

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784,520)
        self.l2 = nn.Linear(520,320)
        self.l3 = nn.Linear(320,240)
        self.l4 = nn.Linear(240,120)
        self.l5 = nn.Linear(120,10)

    def forward(self,x):
        # input data : ( n , 1 , 28 , 28 )
        x = x.view(-1,784) # Flatten : ( n , 784 ) n에서 784로 flatten해야 한다.
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x) #activation X
```

모델을 통해 나온 output값에 실제 target과의 loss를 계산한다.
(CrossEntropyLoss를 통해 criterion 을 선언)

또한, 우리의 train 모델이 적합한지 확인하기 위해, 데이터셋을 train set과 test set으로 나누어 모델을 평가한다.

Train set을 통해 모델을 훈련한 후(train_loader), 사용되지 않은 test 셋을 통해 모델을 평가한다. (test_loader)

```python
def train(epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        optimizer.zero_grad()
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        if batch_idx%50==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss=0
    correct=0
    for data,target in test_loader:

        data = data.to(device)
        target = target.to(device)

        output = model(data)

        test_loss += criterion(output,target).data[0]

        pred = output.data.max(1,keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 9):
    train(epoch)
    test()

```
