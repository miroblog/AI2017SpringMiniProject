i. ���� ������ ���̴� ���� : 
	----- �ۼ��� ���� -----
	[MNIST] mnist_architecture_tuning.py, mnist_objective_func.py
	[Cifar10] cifar10_architecture_tuning.py, cifar10_objective_func.py
	----- ������ �ܺ� ���� -----
	[Cifar10 - data loader] data_helpers.py from https://github.com/wolfib/image-classification-CIFAR10-tf
	
ii. �����ͼ�
	[MNIST] MNIST_data
	[Cifar10 - data]  data_helpers.py from cifar-10-batches-py https://www.cs.toronto.edu/~kriz/cifar.html

iii. ��Ÿ ����, ��� ���

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
 	���� ������ �� �ִ�.	
	
	mnist/cifar10_objective
	* loss function : cross entropy, L2 norm, mean squared error
	�� ������ �� �ִ�.

	(e.g.)
		for n_hidden_layer in [1, 2, 3]: 
			for n_hidden_unit in [10, 50, 100, 200, 400, 800]:
			...
	�����ϰ��� �ϴ� ������ list���ٰ� ä�������� �ȴ�.

	
iv. �ۼ��ڰ� ������ ������ ȯ��
1. ���̽� ���� : Python 3.5.3 
2. �ü�� : Microsoft Windows 10 Pro
3. CPU : Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz, 2601Mhz, 4 �ھ�, 8 �� ���μ���
4. GPU (VRAM) : GeForce GTX 970M
5. RAM : 8.00 GB

**(����) �ټ����� ��� Ȯ��
$ tensorboard --logdir /tmp/17springAI/mnist/
$ tensorboard --logdir /tmp/17springAI/cifar10/


**(����) ���� ��� ���� ���� (�𵨺� : ��Ȯ��, ��ġ �Ʒ� Ƚ��)
mnist_exp1_stat.xlsx : mnist_architecture_tuning.py�� ������ ��� ����
mnist_cifar_exp2_stat.xlsx : mnist_objective_func.py�� cifar10_objective_func.py�� ������ ��� ����
cifar_exp3.xlsx : cifar10_architecture_tuning.py�� ������ ��� ����
