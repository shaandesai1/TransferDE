from distutils.core import setup
setup(
  name = 'transfer_ode',         # How you named your package folder (MyLib)
  packages = ['transfer_ode'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='Harvard',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description='A Python 3 toolset for creating and optimizing Echo State Networks. This library is an extension and expansion of the previous library written by Reinier Maat: https://github.com/1Reinier/Reservoir',# Give a short description about your library
  author = 'Shaan Desai, Hayden Joy',                   # Type in your name
  author_email = 'shaan.desai@some.ox.ac.uk, hnjoy@mac.com',      # Type in your E-Mail
  url = 'https://github.com/blindedjoy/RcTorch-private',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/blindedjoy/RcTorch/archive/v_01.tar.gz',    # I explain this later on
  keywords = ['transfer_learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'torch',
          'argparse',
          'numpy',
          'datetime',
          'pandas',
          'seaborn',
          'matplotlib',
          'torch',
          'torchvision',
          'torchaudio',
          'gpytorch',
          'botorch',
          'ray[default]'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.6',
  ],

)