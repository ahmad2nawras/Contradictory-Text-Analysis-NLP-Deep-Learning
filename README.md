# Contradictory-Text-Analysis-NLP-Deep-Learning

**This project has been done in collaboration with Mohamad Issam Sayyaf**


 We are conducting a classification task on pairs of sentences, which consist of a premise and a hypothesis. The task involves categorizing each pair into one of three categories - entailment, contradiction, or neutral. To illustrate this, let's consider an example using the following premise:

**"He came, he opened the door and I remember looking back and seeing the expression on his face, and I could tell that he was disappointed."**

For the first hypothesis,

**"Just by the look on his face when he came through the door I just knew that he was let down,"**

we can infer that it is true based on the information in the premise. Therefore, this pair is related by entailment.

For the second hypothesis,

**"He was trying not to make us feel guilty but we knew we had caused him trouble,"**

we cannot reach a conclusion based on the information in the premise. Thus, this relationship is neutral.

For the third hypothesis,

**"He was so excited and bursting with joy that he practically knocked the door off its frame,"**

we know that it is untrue as it contradicts the information in the premise. Hence, this pair is related by contradiction.

The dataset contains premise-hypothesis pairs in fifteen different languages, namely Arabic, Bulgarian, Chinese, German, Greek, English, Spanish, French, Hindi, Russian, Swahili, Thai, Turkish, Urdu, and Vietnamese. We are interested only with the English pairs.


<p align="center">
  <img src="https://github.com/IssamSayyaf/Contradictory-Text-Analysis-NLP-Deep-Learning-/blob/main/images/New%20Bitmap%20Image.bmp" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 1: The Dataset Distribution</em>
</p>

Dataset Descriptions:
The dataset provided for Task 2 of the Contradictory Text Analysis competition is a crucial component of the competition. In this task, the dataset contains different couples of sentences (a premise and a hypothesis) in different languages. Each couple has a label of one out of three classes:

- Entailment: the two sentences a related in meaning.
- Neutral: the sentences are neither related nor contraries.
- Contradiction: the sentences are contraries in meaning.

The data set contains two csv files:

- train.csv: This file contains the ID, premise, hypothesis, and label, as well as the language of the text and its two-letter abbreviation
- test.csv: This file contains the ID, premise, hypothesis, language, and language abbreviation, without labels.

The dataset contains a considerable number of samples, among which English samples represent a dominant portion of 56.7% as shown in Figure 1. The Task will develop a model, which will classify only English sentences couples in one of the mentioned classes.

For the English language, the distribution of the three classes in the dataset is nearly uniform, indicating that there is a balanced representation of sentence pairs labeled as Entailment, Neutral, and Contradiction. This suggests that the dataset provides a fair representation of the relationship between sentence pairs in the English language and can be effectively used to train models for the task of Contradictory Text Analysis in this language as shown in Figure 2.

<p align="center">
  <img src="https://drive.google.com/uc?id=1-MWiR2zllSEPMTS2txhmyz8PJmUb29TE" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 2: The distribution every class in English Language</em>
</p>


We have computed some statistical information on the samples in terms of the number of characters present in them. Specifically, we have calculated the mean, standard deviation, and count of characters across all the samples in the dataset as shown in Figure 3.
The mean of characters across all samples provides a measure of the central tendency of the dataset and indicates the typical length of the samples. The standard deviation of characters provides a measure of the dispersion of the data around the mean and indicates how much the lengths of the samples vary from the typical length. The count of characters provides information on the total number of characters present in the dataset.


<p align="center">
  <img src="https://drive.google.com/uc?id=1-N5O45jUOpQHNOOM0ePDyyy4BnNnPwFX" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 3: The statistical study for the dataset.</em>
</p>
These statistics can be useful in understanding the properties of the dataset and in developing models that can handle samples of different lengths effectively. For instance, models that are sensitive to the length of the samples may require additional preprocessing or architecture modifications to handle samples with varying lengths effectively. Overall, the statistical information on the number of characters in the dataset can provide valuable insights for developing effective models for Contradictory Text Analysis. 
 
# Methods
To design an efficient model for Contradictory Text Analysis, we establish a pipeline that consists of the following steps:
## Preprocessing

To design an efficient model for Contradictory Text Analysis, we establish a pipeline that consists of the following steps:

1. **Read the dataset**:
Load the dataset from a CSV file and store it in a panda DataFrame.

2. **Extract English Sentences**:
Extract English sentences from the train and test datasets based on the language's two-letter abbreviation label provided in the CSV files. This will result in a sample size of 6869 sentences for training.

3. **Split the Dataset**:
Split the data into training, validation, and test sets with 70%, 25%, and 5% of the data, respectively.

4. **Tokenize the text**:
In natural language processing (NLP), a tokenizer is a tool that splits text into smaller chunks called tokens. These tokens can be individual words or groups of words that are meaningful together, such as named entities or phrases.

   We use the Tokenizer class provided by TensorFlow's Keras API as a text-preprocessing tool. It converts text into sequences of integers that can be fed into a neural network for further processing.

By following this preprocessing pipeline, we can prepare the Contradictory Text Analysis dataset and proceed with designing and training the model architecture. After training, we can evaluate the performance of the model on the dataset. We use RobertaTokenizer which is a tokenizer class provided by the Hugging Face Transformers library, is used to convert text data into numerical sequences that can be fed into a deep learning model such as BERT, RoBERTa, or other transformer-based models. It is specifically designed to work with the RoBERTa model architecture and is based on the Byte-Pair Encoding (BPE) algorithm.
The tokenizer concatenates both premise and hypothesis sentences and tokenizes them together enabling both truncation and padding of the input sequences to ensure that they have the same length.
This code creates a Dataset object from the training encodings and labels. It first creates a tuple of (input_ids, attention_mask) dictionary and the corresponding labels. Finally, the batch method batches the dataset with a specified batch size BATCH_SIZE. The resulting train_dataset and val_dataset is a TensorFlow dataset object that can be used for training the model.

# Designing and Training the Model Architecture

The model architecture uses the RoBERTa language model, which is a state-of-the-art language model based on the Transformer architecture. It is an improvement over the original BERT model, developed by Facebook AI in 2019, and has shown better generalization and higher performance on downstream NLP tasks.

The RoBERTa model uses the same architecture as BERT as shown in Figure 4, consisting of an encoder composed of several Transformer blocks. Each Transformer block consists of a self-attention mechanism and a feedforward neural network.

The key innovation in the Transformer architecture is the self-attention mechanism, which allows the model to weigh the importance of different words in a sequence when generating the output. The self-attention mechanism is applied to both the input (encoder) and output (decoder) sequences, allowing the model to take into account the entire context of the input sequence when generating the output.

<p align="center">
  <img src="https://github.com/IssamSayyaf/Contradictory-Text-Analysis-NLP-Deep-Learning-/blob/main/images/The%20architecture%20RoBERTa%20model.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 4: The architecture RoBERTa model.</em>
</p>



The RoBERTa model is available in several sizes ranging from small to large, with varying numbers of layers and hidden units. In this model architecture, the pre-trained RoBERTa base model is used, which has 12 transformer layers.



The model takes two inputs:

1. **Input_ids**: This refers to the tokenized input sequence where each token is mapped to its corresponding token ID in the RoBERTa vocabulary. During tokenization, the input sequence is first split into words, and each word is then split into sub-words using Byte Pair Encoding (BPE). Each sub-word is then mapped to its corresponding token ID using the RoBERTa tokenizer's vocabulary.

2. **Attention_mask**: In natural language processing (NLP), the attention mechanism is used to weigh the importance of each word in a sentence when processing the sentence. The attention mask is a binary mask used to indicate which tokens in the input sequence should be attended to by the model, and which tokens should be ignored.
   
These inputs are then fed into the RoBERTa model, which is pre-trained with weights from a large corpus of text to learn general language patterns and features. The model then outputs a sequence of hidden states, one for each input token.
The RoBERTa model's architecture with its 12 transformer layers allows it to effectively capture contextual information and semantic representations from the input text, enabling it to perform well on various downstream NLP tasks.

The output of the RoBERTa model is passed through two parallel layers: global_average_pooling and global_max_pooling. Both operations reduce the spatial dimensions of the feature maps and produce a fixed-length vector for each feature map. These vectors are then concatenated and fed into two fully connected dense layers, each with a dropout rate of 0.3, to prevent overfitting. The first dense layer has 64 units, and the last dense layer has three units, which is equal to the number of classification classes. The output of the last dense layer is passed through a softmax activation function, which outputs a probability distribution over the three classes.

The model is trained using a cross-entropy loss function and optimized using the Adam optimizer. During training, the weights of the RoBERTa model are fine-tuned on a smaller labeled dataset specific to the given NLP task. This process adapts the general language model to the specific task, resulting in better performance on the task.

Overall, the model architecture is shown by Figure 5, takes advantage of the state-of-the-art RoBERTa language model and fine-tuning techniques to achieve high performance on the classification task. The self-attention mechanism in the Transformer architecture allows the model to take into account the entire context of the input sequence, making it suitable for handling long sequences. The parallel layers and fully connected dense layers provide a way to reduce the spatial dimensions of the feature maps and perform classification on the fixed-length vectors. The use of dropout layers and the softmax activation function help prevent overfitting and produce a probability distribution over the classes.

<p align="center">
  <img src="https://github.com/IssamSayyaf/Contradictory-Text-Analysis-NLP-Deep-Learning-/blob/main/images/The%20Architecture%20of%20the%20proposed%20model.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 5: The Architecture of the proposed model.</em>
</p>
## Results

After training the model with different configurations, we evaluated the model's performance using the test dataset. The results of the experiments are summarized in the following table (Table 1).

**Table 1: Model Performance with Different Configurations**

| Configuration | Optimizer          | Learning Rate | Epsilon  | Accuracy |
|--------------|--------------------|---------------|----------|----------|
| 1            | Adam               | 1e-7          | 1e-8     | 82%      |
| 2            | RMSprop            | 1e-6          | 1e-8     | 78%      |
| 3            | SGD                | 1e-6          | 1e-8     | 71%      |

As we can see from the table, the best configuration was the first one, which used the Adam optimizer with a learning rate of 1e-7 and an epsilon of 1e-8. This configuration achieved an accuracy of 82% on the test dataset (see Figure 6).

The second configuration, which used the RMSprop optimizer with a learning rate of 1e-6 and an epsilon of 1e-8, achieved an accuracy of 78% on the test dataset. Meanwhile, the third configuration, which used the SGD optimizer with a learning rate of 1e-6 and an epsilon of 1e-8, achieved an accuracy of 71% on the test dataset.

During the training process, we utilized two important callbacks to enhance the model's performance:

1. **EarlyStopping:** EarlyStopping stopped the training process if there was no improvement in the validation accuracy after a certain number of epochs. This helped prevent unnecessary computations and overfitting.

2. **ReduceLROnPlateau:** ReduceLROnPlateau reduced the learning rate when the validation loss stopped improving. This adaptive learning rate adjustment further improved the model's convergence and performance.

Overall, the results highlighted the significance of the optimizer and learning rate choices in determining the model's performance. Using the Adam optimizer with a lower learning rate and epsilon achieved the best accuracy. Additionally, the utilization of EarlyStopping and ReduceLROnPlateau callbacks contributed to preventing overfitting and enhancing the overall model performance.

<p align="center">
  <img src="https://github.com/IssamSayyaf/Contradictory-Text-Analysis-NLP-Deep-Learning-/blob/main/images/Training%20Progress%20Evolution%20of%20Accuracy%20and%20Loss%20over%20Epochs.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 6: Training Progress Evolution of Accuracy and Loss over Epochs.</em>
</p>

To evaluate the performance of the model, we analyzed the confusion matrix on the test dataset. The confusion matrix reveals the number of true positives, true negatives, false positives, and false negatives for each class in the classification task. From the confusion matrix, we can calculate the recall, precision, and F1 score for each class. The confusion matrix for our model on the test dataset is shown in Figure 7.

<p align="center">
  <img src="https://github.com/IssamSayyaf/Contradictory-Text-Analysis-NLP-Deep-Learning-/blob/main/images/Confusion%20Matrix%20Evaluation%20of%20Model%20Performance%20through%20Actual%20vs%20Predicted%20Classifications.png" alt="alt text" width="width" height="height" />
  <br>
  <em>Figure 7: Confusion Matrix Evaluation of Model Performance through Actual vs Predicted Classifications.</em>
</p>

We notice that the model has the highest accuracy in predicting Contradiction sentences with an accuracy of 83.75%. Entailment and Neutral sentences, on the other hand, have accuracies of 78.5% and 75%, respectively. Additionally, we can see from the confusion matrix that the model tends to make more false positives for Neutral and Contradiction sentences compared to Entailment sentences.
