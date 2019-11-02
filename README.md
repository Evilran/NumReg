# NumReg: Handwritten Numbers Recognition
[![Python 3.5](https://img.shields.io/badge/python-3.5-yellow.svg)](https://www.python.org/)[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Evilran/NumReg)

    __  __                      ____
    /\ \/\ \                    /\  _`\
    \ \ `\ \  __  __    ___ ___\ \ \L\ \     __     __
    \ \ , ` \/\ \/\ \ /' __` __`\ \ ,  /   /'__`\ /'_ `\
    \ \ \`\ \ \ \_\ \/\ \/\ \/\ \ \ \ \ /\  __//\ \L\ \
    \ \_\ \_\ \____/\ \_\ \_\ \_\ \_\ \_\ \____\ \____ \
        \/_/\/_/\/___/  \/_/\/_/\/_/\/_/\/ /\/____/\/___L  |
                                                    /\___//
                                                    \_/__/
    
                                    --version:1.0 Evi1ran


Requirements
------------------------------------------------------------------

- pandas==0.24.1

- Pillow==6.2.0

- scikit_learn==0.21.3

  


Usage
---
```
optional arguments:
  -h, --help            show this help message and exit
  -g G                  generate handwritten numbers database (default generate 10000 pics)
  -r R                  recognize handwritten number
  -s S                  handwritten numbers database source font (default font is lib/msyh.ttc)
  -f {1,2,3,4}, --flag {1,2,3,4}
                        1. LR，Logistic Regression
                        2. Linear SVC，Support Vector Classification
                        3. MLPC，Multi-Layer Perceptron Classification
                        4. SGDC，Stochastic Gradient Descent Classification
```



Here's a simple example of how to recognize handwritten number:

```
$ python3 NumReg.py -r test_1.png  
```

It uses the first method by default, which is *LR* (Logistic Regression), If you want to change its method, you can add the argument `-f` to the command. Just like using the fourth method (SGDC):

```
$ python3 NumReg.py -r test_1.png -f 4
```



In addition, you are allowed to generate your own data sets and train your model in this way:

```
$ python3 NumReg.py -g 10000
```

The dataset generates 10,000 images by default, of course you can modify this parameter. You can even modify the font used to create the training set:

```
$ python3 NumReg.py -g 10000 -s msyh.ttc
```

The default font is the **Microsoft YaHei** font in the  `/lib` .

