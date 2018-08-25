i. 실험 재현에 쓰이는 파일 : 
	----- 작성한 파일 -----
	[MNIST] mnist_architecture_tuning.py, mnist_objective_func.py
	[Cifar10] cifar10_architecture_tuning.py, cifar10_objective_func.py
	----- 참고한 외부 파일 -----
	[Cifar10 - data loader] data_helpers.py from https://github.com/wolfib/image-classification-CIFAR10-tf
	
ii. 데이터셋
	[MNIST] MNIST_data
	[Cifar10 - data]  data_helpers.py from cifar-10-batches-py https://www.cs.toronto.edu/~kriz/cifar.html

iii. 기타 실행, 사용 방법

	$ python3 mnist_architecture_tuning.py
	$ python3 mnist_objective_func.py
	$ python3 cifar10_architecture_tuning.py 
	$ python3 cifar10_objective_func.py

	mnist/cifar10 _architecture_tuning.py :  
	* hidden layer, unit 
	* learning rate
	* activation function
	* dropout - rate
	* batch normalisation
 	등을 조정할 수 있다.	
	
	mnist/cifar10_objective
	* loss function : cross entropy, L2 norm, mean squared error
	를 조정할 수 있다.

	(e.g.)
		for n_hidden_layer in [1, 2, 3]: 
			for n_hidden_unit in [10, 50, 100, 200, 400, 800]:
			...
	실험하고자 하는 설정을 list에다가 채워넣으면 된다.

	
iv. 작성자가 실험을 수행한 환경
1. 파이썬 버전 : Python 3.5.3 
2. 운영체제 : Microsoft Windows 10 Pro
3. CPU : Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 2601Mhz, 4 코어, 8 논리 프로세서
4. GPU (VRAM) : GeForce GTX 970M
5. RAM : 8.00 GB

**(참조) 텐서보드 결과 확인
$ tensorboard --logdir /tmp/17springAI/mnist/
$ tensorboard --logdir /tmp/17springAI/cifar10/


**(참조) 실험 결과 엑셀 파일 (모델별 : 정확도, 배치 훈련 횟수)
mnist_exp1_stat.xlsx : mnist_architecture_tuning.py의 정리된 결과 파일
mnist_cifar_exp2_stat.xlsx : mnist_objective_func.py와 cifar10_objective_func.py의 정리된 결과 파일
cifar_exp3.xlsx : cifar10_architecture_tuning.py의 정리된 결과 파일
