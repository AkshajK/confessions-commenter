import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="confessionscommenter", # Replace with your own username
    version="0.0.9",
    author="Raul Alcantara, Erick Gbordzoe, Akshaj Kadaveru, Sean Elliott, Helen Lu",
    author_email="shareconfessionscommenter@gmail.com",
    description="Autogenerate Comments for Facebook Posts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AkshajK/confessionscommenter",
    project_urls={
        "Bug Tracker": "https://github.com/AkshajK/confessionscommenter/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=[
        'rich',
        'facebook-scraper',
        'pyperclip',
        'transformers',
        'tensorflow',
        'nltk',
        'pillow',
        'pywin32; platform_system=="Windows"'

    ],
    package_data ={"confessionscommenter": ["./meme_data.json"]},
    # package_data  = [("confessionscommenter", ["src/confessionscommenter/meme_data.json"])],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
