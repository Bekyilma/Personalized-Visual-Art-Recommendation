# Personalised Visual Art Recommendation by Learning Latent Semantic Representations
<p align="center">
<img width="1100"  src="/figures/cover.jpg"/> 
</p>

# Introduction
<p align="justify">
In Recommender systems, data representation techniques play a great role as they have the power to entangle, hide and reveal explanatory factors embedded within datasets. Hence, they influence the quality of recommendations. Specifically, in Visual Art (VA) recommendations the complexity of the concepts embodied within paintings, makes the task of capturing semantics by machines far from trivial. In VA recommendation, prominent works commonly use manually curated metadata to drive recommendations. Recent works in this domain aim at leveraging visual features extracted using Deep Neural Networks (DNN). However, such data representation approaches are resource demanding and do not have a direct interpretation, hindering user acceptance. To address these limitations, this work proposes an approach for Personalised Recommendation of Visual arts based on learning latent semantic representation of paintings. This is done by training a Latent Dirichlet Allocation (LDA) model on textual descriptions of paintings. The trained LDA model manages to successfully uncover non-obvious semantic relationships between paintings whilst being able to offer explainable recommendations. Experimental evaluations demonstrate that our method tends to perform better than exploiting visual features extracted using pre-trained Deep Neural Networks. 
</p>


## Requirements

<table>
<tr>
  <td>NumPy</td>
  <td>
    <a href="https://www.numpy.org/">
    <img src="https://img.shields.io/badge/NumPy-v1.19.1-green" alt="NumPy" />
    </a>
  </td>
</tr>
<tr>
  <td>SciPy</td>
  <td>
    <a href="https://www.scipy.org/">
    <img src="https://img.shields.io/badge/SciPy-v1.5.2-red" alt="SciPy" />
    </a>
  </td>
</tr>
<tr>
  <td>scikit-learn</td>
  <td>
    <a href="https://www.scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-v0.23.2-blueviolet" alt="scikit-learn" />
    </a>
</td>
</tr>
<tr>
  <td>Pandas</td>
  <td>
    <a href="https://www.pandas.pydata.org/">
    <img src="https://img.shields.io/badge/pandas-v1.1.1-blue" alt="Pandas" />
    </a>
  </td>
</tr>
<tr>
  <td>Matplotlib</td>
  <td>
    <a href="https://matplotlib.org/">
    <img src="https://img.shields.io/badge/Matplotlib-v3.3.1-orange" alt="Matplotlib" />
    </a>
  </td>
</tr>
<tr>
	<td>gensim</td>
	<td>
		<a href="https://radimrehurek.com/gensim/">
		<img src="https://img.shields.io/badge/gensim-v3.8.3-blue"  alt="gensim" />
	</a>
	</td>
</tr>
<tr>
	<td>spaCy</td>
	<td>
		<a href="https://spacy.io/usage">
		<img src="https://img.shields.io/badge/spaCy-v2.3.2-ff69b4"  alt="spaCy" />
	</a>
	</td>
</tr>
</table>

This code  works on Python 3.5 or later.

* * *
# Usage
The Painting_LDA model trained on the National Gallery dataset can be found in [/resources/models/](https://github.com/Bekyilma/Visual_art-recommender/tree/master/resources/models)

If you want to train with your own painting descrtiption corpus:
> use the script [text-cleaning.py](https://github.com/Bekyilma/Visual_art-recommender/blob/master/text-cleaning.py) to clean your corpus. It will save the pre-processed data in [resources/datasets/preprocessed/](https://github.com/Bekyilma/Visual_art-recommender/tree/master/resources/datasets/preprocessed)

> After claning your courpus use the script [lda-training.py](https://github.com/Bekyilma/Visual_art-recommender/blob/master/lda-training.py) to train the Painting-LDA model. 

To make recommendation use the script [query-lda.py](https://github.com/Bekyilma/Visual_art-recommender/blob/master/query-lda.py)

> It generates a list of recommendations that are the most similar to a list of paintings liked or rated by a user.

Citation
========

When you use this work or method for your research, we ask you to cite the following publication:

[Bereket Abera Yilma, Najib Aghenda, Marcelo Romero, Yannick Naudet and Hervé Panetto (2020), Personalized Visual Art Recommendation by Learning Latent Semantic Representations in Semantic and Social Media Adaptation & Personalization doi:10.1109/SMAP49528.2020.9248448](https://ieeexplore.ieee.org/abstract/document/9248448)

