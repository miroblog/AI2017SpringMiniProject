# Mini Project for Spring17 A.I. Course

### Experiment with MLP
* `# Layers, # Neurons`
* `L2 normalization`
* `Dropout`
* `BatchNormalization`
* `Objective Functions`
* `Activation Functions`

For Detailed Result see... [Report](https://github.com/miroblog/AI2017SpringMiniProject/blob/master/AI17S_Report.pdf)  
[Code]  
* [MNIST] mnist_architecture_tuning.py, mnist_objective_func.py
* [Cifar10] cifar10_architecture_tuning.py, cifar10_objective_func.py   

## How to run ...

```
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
	or n_hidden_layer in [1, 2, 3]: 
	or n_hidden_unit in [10, 50, 100, 200, 400, 800]:
			...
실험하고자 하는 설정을 list에다가 채워넣으면 된다.

```

### Prerequisites

tensorflow, sklearn, nuumpy, pandas ...

```python
pip install -r requirements.txt
```

## Authors

* **Lee Hankyol** - *Initial work* - [17AIMiniProject](https://github.com/miroblog/AI2017SpringMiniProject)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
