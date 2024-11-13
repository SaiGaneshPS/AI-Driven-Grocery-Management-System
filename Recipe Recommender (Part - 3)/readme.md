The dataset is split into 5 parts and trained on 5 different notebooks. The GPT-2 model is fine-tuned on this dataset to improve it's adaptability and performance to the scenario.

* GPT_1 : Trained on the most common ingredients
* GPT_2 : 0th Row - 57909th Row
* GPT_3 : 57909th Row - 115818th Row
* GPT_4 : 115818th Row - 173727th Row
* GPT_5 : 173727th Row - 231637th Row

Additionaly, the approach of adding tokens in the beginning of each field during the training process has proved to improve the performance greatly.
The model is trained on inputs containing **Cooking time** **Ingredients** and **Steps**, After adding tokens the input looks like:

><start-time> 22 minutes <end-time> <start-ingredients> white rice <sep> water <sep> french vanilla instant pudding <sep> evaporated milk <sep> raisin <sep> nutmeg <end-ingredients> <start-steps> ['in medium sauce pan combine water and rice ,… '] <end-steps>

The model when given **Cooking time** and **Ingredients** produces an output like:

![image](https://github.com/user-attachments/assets/8bc28746-44f6-4865-95a8-cc43ad1905cf)

Which then requires formatting to remove the tokens and paranthesis to produce clean steps to be followed.
