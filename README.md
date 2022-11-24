# Implementing Text Summarization using Deep Learning and Python

## Roshin Nishad AI Project. 2022©
### 2020BCS0019

Customer reviews can often be long and descriptive. Analyzing these reviews manually, as you can imagine, is really time-consuming. This is where the brilliance of Natural Language Processing can be applied to generate a summary for long reviews.

Let’s first understand what text summarization is before we look at how it works. Here is a succinct definition to get us started:

“Automatic text summarization is the task of producing a concise and fluent summary while preserving key information content and overall meaning”

    -Text Summarization Techniques: A Brief Survey, 2017

There are broadly two different approaches that are used for text summarization:

- Extractive Summarization
- Abstractive Summarization

### Extractive Summarization
The name gives away what this approach does. We identify the important sentences or phrases from the original text and extract only those from the text.

### Abstractive Summarization
This is a very interesting approach. Here, we generate new sentences from the original text. This is in contrast to the extractive approach we saw earlier where we used only the sentences that were present.

### Some outputs from running the code ---

`data['Text'][:10]`

![image](https://user-images.githubusercontent.com/3417276/203812786-457e67dc-da02-46e2-8bcb-b9ee1a9e0a6b.png)


`data['Summary'][:10]`

![image](https://user-images.githubusercontent.com/3417276/203812965-21b5c60c-09f4-4af0-92a9-37fd600de402.png)


`for i in range(5):
print("Review:",data['cleaned_text'][i])
print("Summary:",data['cleaned_summary'][i])
print("\n")`

![image](https://user-images.githubusercontent.com/3417276/203813075-3e30a4bc-da4d-4d8c-9efe-ffe58eeb9604.png)


`length_df = pd.DataFrame({'text':text_word_count, 'summary':summary_word_count})
length_df.hist(bins = 30)
plt.show()`

![image](https://user-images.githubusercontent.com/3417276/203813170-006cd847-ae1b-489f-bc62-c61a01702308.png)

`model.summary()`

![image](https://user-images.githubusercontent.com/3417276/203813281-321552e1-3119-475c-9850-b5a7505c5225.png)


`history=model.fit([x_tr,y_tr[:,:-1]], y_tr.reshape(y_tr.shape[0],y_tr.shape[1], 1)[:,1:],epochs=2,callbacks=[es],batch_size=1000, validation_data=([x_val,y_val[:,:-1]], y_val.reshape(y_val.shape[0],y_val.shape[1], 1)[:,1:]))`

![image](https://user-images.githubusercontent.com/3417276/203813423-a1a268ce-862e-4faf-8920-067aa4beeeb8.png)


`pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()`

![image](https://user-images.githubusercontent.com/3417276/203813488-2cce5959-3636-49ec-86d8-820e0492a56d.png)


`for i in range(30,100):
print("Original Human-made Review:",seq2text(x_tr[i]))
print("-------------Summary Below-------------")
print("Predicted summary:",seq2summary(y_tr[i]))
print("\n\n")`

![image](https://user-images.githubusercontent.com/3417276/203813591-8761eee1-0454-4c25-97b4-ffcb6b182ef1.png)


### Thank You.
## Roshin Nishad 2022©
## 2020BCS0019.

MIT Licensed.





