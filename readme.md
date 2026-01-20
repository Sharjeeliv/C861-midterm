## **Comparing Deep Learning Models and Cross-Script Transfer Learning between Arabic and Urdu Handwritten Letters**  
Author: Sharjeel Mustafa

### **Introduction**  
As research into artifical intelligence, particularly natural language processing, continues to expand, languages that are central to the production of research (e.g., English) are better suited to take advantage of such developments, leaving underdeveloped languages vulnerable. In this study we implement a variety models for handwritten letter recognition on English, Arabic, and Urdu, with models like CNN undergoing architectural tuning. 

After training, we examine the feasability of using same-script languages to enhance the performance of similar languages with limited to language-specific data and models. We examine English (latin-based alphabet) against Arabic and Urdu (arabic-based alphabet) to determine if languages belonging to the same script can leverage that fact. This has the potential implications that a dominant script-language can be used to rapidly develop artificial intelligence for same-script languages with limited resources.


### **Installation**  
Please ensure the AHCD, UHAT, and EMNIST datasets are downloaded and placed in the data folder with abbreviations listed in the paper. Some utils are available in data.py but require manual execution to preprocess the data. Once datasets are prepared and the conda environment is setup, the code can be executed (by first altering the main.py depending on the task) with the following command.

```py
python src.main
```

### **Citation**
If you find this work useful, please consider citing:
```
@misc{sharjeelc861m,
  title={Comparing Deep Learning Models and Cross-Script Transfer Learning between Arabic and Urdu Handwritten Letters},
  author={Sharjeel Mustafa},
  year={2025}
}
```

**Errata**
```
Eq 2: should be ...(i,j)=\sum_M...
```
