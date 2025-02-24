from distutils.core import setup
setup(
    name='perceptual-advex',
    packages=[
        'perceptual_advex',
    ],
    package_data={'perceptual_advex': ['py.typed']},
    version='0.2.6',
    license='MIT',
    description='Code for the ICLR 2021 paper "Perceptual Adversarial Robustness: Defense Against Unseen Threat Models"',
    author='Cassidy Laidlaw',
    author_email='claidlaw@umd.edu',
    url='https://github.com/cassidylaidlaw/perceptual-advex',
    download_url='https://github.com/cassidylaidlaw/perceptual-advex/archive/TODO.tar.gz',
    keywords=['adversarial examples', 'machine learning'],
    install_requires=[
        'torch>=1.4.0',
        'robustness>=1.1.post2',
        'numpy>=1.18.2',
        'torchvision>=0.5.0',
        'PyWavelets>=1.0.0',
        'advex-uar>=0.0.5.dev0',
        'statsmodels==0.11.1',
        'recoloradv==0.0.1',
        'advfussion'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
) 
