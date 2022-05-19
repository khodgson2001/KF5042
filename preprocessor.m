%CREDITS: Code modified from KF5042 workshops

function [documents] = preprocessor(textData) % Convert the text data to lowercase.
cleanTextData = lower(textData);
% Tokenize the text.
%documents = tokenizedDocument(cleanTextData); % Erase punctuation.
documents = erasePunctuation(cleanTextData);
documents = tokenizedDocument(documents);
% Remove a list of stop words.
documents = removeStopWords(documents); 
end