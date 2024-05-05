import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

    setuptools.setup(
        name='VisionForge',
        version='1.0.0',
        packages=['VisionForge'],
        url='https://github.com/Thigos/VisionForge',
        license='AGPL-3.0 license',
        author='Thiago Rodrigues',
        author_email='',
        description='Tracking é um projeto de visão computacional que otimiza o rastreamento de objetos. Combina '
                    'YOLOv8 para detecção em intervalos programados, reduzindo o custo computacional, e OpenCV para '
                    'rastreamento em tempo real. Uma solução eficiente e econômica para monitoramento de objetos.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        project_urls={
            'Documentation': 'https://github.com/Thigos/VisionForge/blob/main/README.md',
            'Bug Reports':
                'https://github.com/Thigos/VisionForge/issues',
            'Source Code': 'https://github.com/Thigos/VisionForge'
        },
        classifiers=[
            # see https://pypi.org/classifiers/
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Software Development :: Build Tools',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
            'Operating System :: OS Independent',
        ],
        python_requires='>=3.7',
        install_requires=['ultralytics~=8.0.118',
                          'opencv-python~=4.6.0.66',
                          'numpy~=1.23.5',
                          'torch>=1.8.1']
    )
