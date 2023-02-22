import setuptools

with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setuptools.setup(name='little_RNN',
                packages=['data', 'rnn', 'controller'],
                install_requires=install_requires)
