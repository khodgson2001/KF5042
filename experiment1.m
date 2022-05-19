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

%load in dataset, get assigned score and sanitise userReviews
reviews = readtable('trainingData.csv', 'TextType','string');
userScore = reviews.user_suggestion;
userReviews = preprocessor(reviews.user_review);


sentiment = zeros(size(userReviews));

%for every review, get the words contained and compare to documented words
%if the word is present in the dictionary get the sentiment score of the
%word and add it to total score
for i = 1:userReviews.length
    word2cmp = userReviews(i).Vocabulary;
    for j = 1:length(word2cmp)
        if words_hash.containsKey(word2cmp(j))
            sentiment(i) = sentiment(i) + words_hash.get(word2cmp(j));
        end
    end

    %normalise sentiment score so -1 given for negative sentiment and +1
    %for positive sentiment
    if (sentiment(i)>=1)
        sentiment(i)=1;
    elseif (sentiment(i)<=-1)
        sentiment(i) = -1;
    end
end

fprintf('Processed %d reviews, with %s words, my score was %d and the true score was %d\n', i, joinWords(userReviews(i)), sentiment(i), userScore(i));

%total number of all 0 rated sentiments (either neutral or not found
nonVal = sum(sentiment == 0);

%total distinct sentiments
dist_sentiment = numel(sentiment) - nonVal;

fprintf('Total coverage of positive and negative classes: %2.2f%%, total distinct values: %d, NaN values: %d\n',(dist_sentiment * 100)/numel(sentiment), dist_sentiment, nonVal);

%calculate true positive, true negative and accuracy as a percentage
tPos = sentiment((sentiment==1) & (userScore==1));
tNeg = sentiment((sentiment==-1) & (userScore==0));
accuracy = (numel(tPos) + numel(tNeg)) * 100 / dist_sentiment;


%output accuracy true pos true neg
fprintf("Accuracy: %2.2f%%, True positive: %d, True negative: %d\n", accuracy, numel(tPos), numel(tNeg)); 
figure %confusion matrix
confusionchart(userScore, sentiment);
