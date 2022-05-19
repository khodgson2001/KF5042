%CREDITS: Code modified from KF5042 workshops

%clean workspace
clc;
clear;

%open positive and negative dicts
opPos = fopen(fullfile('opinion-lexicon-English', 'positive-words.txt'));
opNeg = fopen(fullfile('opinion-lexicon-English', 'negative-words.txt'));

%remove comments in files and convert from array to string
opPos = textscan(opPos, '%s', 'CommentStyle', ';');
opPos = string(opPos{1});

opNeg = textscan(opNeg, '%s', 'CommentStyle', ';');
opNeg = string(opNeg{1});

%close open files
fclose all;

%allow use of java hashtables
words_hash = java.util.Hashtable;

%create hashtable and insert all positive words into it
[opPosSize, ~] = size(opPos);
for i = 1:opPosSize
       words_hash.put(opPos(i, 1), 1);
end

%create hashtable and insert all negative words
[opNegSize, ~] = size(opNeg);
for i = 1:opNegSize
        words_hash.put(opNeg(i, 1), 1);
end

%load in fastText toolbox
rng('default');
emb = fastTextWordEmbedding;

%create a table containing labelled words
words=[opPos;opNeg];
labels = categorical(nan(numel(words),1));
labels(1:numel(opPos)) = "Positive";
labels(numel(opPos)+1:end) = 'Negative';
data = table(words, labels, 'VariableNames', {'Word','Label'});

%remove any words not contained in word embeddings that are in Lius lexicon
idx = ~isVocabularyWord(emb, data.Word);
data(idx,:) = [];

%get the total number of words
totalWords = size(data,1);
%split the data into training and test data with a 90:10 ratio
%90% for training, 10% for testing
cvp = cvpartition(totalWords,'HoldOut',0.01); 
dataTrain = data(training(cvp),:); dataTest = data(test(cvp),:);
%convert word in training data to vectors using word2vec
wordsTrain = dataTrain.Word; 
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

%train SVM classifier into positive and negative categories
model = fitcsvm(XTrain,YTrain);

% Test Trained Model 
wordsTest = dataTest.Word;
    XTest = word2vec(emb,wordsTest);
    YTest = dataTest.Label;
%Predict the sentiment labels of the test word vectors. 
[YPred,scores] = predict(model,XTest);
%Visualize the confusion matrix.
figure
    confusionchart(YTest,YPred);


%load in dataset, get assigned score and sanitise userReviews
reviews = readtable('trainingData.csv', 'TextType','string');
userScore = reviews.user_suggestion;
userReviews = preprocessor(reviews.user_review);

%idx = ~isVocabularyWord(emb,sents.Vocabulary); %18b
idx = ~ismember(emb,userReviews.Vocabulary); %18a
removeWords(userReviews, idx);

sentiment = zeros(size(userReviews));

%for every review, get the sentence contained and conver it to a vector,
%and then predict the sentiment scored of the vectors using the previous
%model, giving the mean score at the end
for i = 1:userReviews.length
    docwords = userReviews(i).Vocabulary;
    vec = word2vec(emb,docwords);
    [~,scores] = predict(model,vec);
    sentiment(i) = mean(scores(:,1));
    if isnan(sentiment(i))
        sentiment(i) = 0;
    end
end

fprintf('Processed %d reviews, with [%s] words, my score was %d and the true score was %d\n', i, joinWords(userReviews(i)), sentiment(i), userScore(i));

%total number of all 0 rated sentiments (either neutral or not found
nonVal = sum(sentiment == 0);

%total distinct sentiments
dist_sentiment = numel(sentiment) - nonVal;

fprintf('Total coverage of positive and negative classes: %2.2f%%, total distinct values: %d, NaN values: %d\n',(dist_sentiment * 100)/numel(sentiment), dist_sentiment, nonVal);

confusionchart(userScore, sentiment);
