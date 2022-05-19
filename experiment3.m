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

rng('default');
emb = fastTextWordEmbedding;

words=[opPos;opNeg];
labels = categorical(nan(numel(words),1));
labels(1:numel(opPos)) = "Positive";
labels(numel(opPos)+1:end) = 'Negative';
data = table(words, labels, 'VariableNames', {'Word','Label'});

idx = ~isVocabularyWord(emb, data.Word);
data(idx,:) = [];

totalWords = size(data,1);
cvp = cvpartition(totalWords,'HoldOut',0.01); 
dataTrain = data(training(cvp),:); dataTest = data(test(cvp),:);
wordsTrain = dataTrain.Word; 
XTrain = word2vec(emb,wordsTrain);
YTrain = dataTrain.Label;

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

%for every review, get the words contained and compare to documented words
%if the word is present in the dictionary get the sentiment score of the
%word and add it to total score
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

sentiment = zeros(size(userReviews));

for ii = 1 : userReviews.length
    docwords = userReviews(ii).Vocabulary;
    for jj = 1 : length(docwords)
        if words_hash.containsKey(docwords(jj))
            sentiment(ii) = sentiment(ii) +  words_hash.get(docwords(jj));
        end
    end
    if sentiment(ii) == 0
        vec = word2vec(emb,docwords);
        [~,scores] = predict(model,vec);
        sentiment(ii) = mean(scores(:,1));
        if isnan(sentiment(ii))
            sentiment(ii) = 0;
        end
    end
    if sentiment(ii) ~= 0
        fprintf('+++Sent: %d, words: %s, FoundScore: %d, GoldScore: %d\n', ii, joinWords(userReviews(ii)), sentiment(ii), userScore(ii));
    else
        fprintf('---Sent: %d, words: %s, Not Covered, GoldScore: %d\n', ii, joinWords(userReviews(ii)),  userScore(ii));
    end
end

fprintf('Processed %d reviews, with [%s] words, my score was %d and the true score was %d\n', i, joinWords(userReviews(i)), sentiment(i), userScore(i));

%total number of all 0 rated sentiments (either neutral or not found
nonVal = sum(sentiment == 0);

%total distinct sentiments
dist_sentiment = numel(sentiment) - nonVal;

fprintf('Total coverage of positive and negative classes: %2.2f%%, total distinct values: %d, NaN values: %d\n',(dist_sentiment * 100)/numel(sentiment), dist_sentiment, nonVal);

confusionchart(userScore, sentiment);
