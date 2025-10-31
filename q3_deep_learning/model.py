import torch
import torch.nn as nn
import torch.nn.functional as F

class WordMuncher(nn.Module):
    def __init__(self, dictionarySize, sparkleFactor, brainSize, howManyLayers, brainCellType='GRU', forgetfulness=0.1):
        super(WordMuncher, self).__init__()
        
        self.embedding = nn.Embedding(dictionarySize, sparkleFactor)
        
        rnn_cell = getattr(nn, brainCellType)
        
        self.rnn = rnn_cell(sparkleFactor, brainSize, howManyLayers, 
                            dropout=forgetfulness, batch_first=True, 
                            bidirectional=True)
        
        self.thoughtSqueezer = nn.Linear(brainSize * 2, brainSize)
        self.memorySqueezer = nn.Linear(brainSize * 2, brainSize)
        self.brainCellType = brainCellType

    def forward(self, wordsIn):
        shinyWords = self.embedding(wordsIn)
        
        allTheThoughts, lastThought = self.rnn(shinyWords)
        
        if self.brainCellType == 'LSTM':
            thoughtVector, memoryVector = lastThought
            thoughtVector = thoughtVector.view(self.howManyLayers, 2, -1, self.brainSize).permute(0, 2, 1, 3).contiguous()
            memoryVector = memoryVector.view(self.howManyLayers, 2, -1, self.brainSize).permute(0, 2, 1, 3).contiguous()
            
            thoughtVector = self.thoughtSqueezer(thoughtVector.view(self.howManyLayers, -1, self.brainSize * 2))
            memoryVector = self.memorySqueezer(memoryVector.view(self.howManyLayers, -1, self.brainSize * 2))
            lastThought = (thoughtVector, memoryVector)
        else:
            lastThought = lastThought.view(self.howManyLayers, 2, -1, self.brainSize).permute(0, 2, 1, 3).contiguous()
            lastThought = self.thoughtSqueezer(lastThought.view(self.howManyLayers, -1, self.brainSize * 2))
            
        return allTheThoughts, lastThought

class StaringContest(nn.Module):
    def __init__(self, brainSize):
        super(StaringContest, self).__init__()
        self.brainSize = brainSize

    def forward(self, bakersThought, allTheThoughts):
        starePoints = torch.sum(allTheThoughts, dim=2)
        return F.softmax(starePoints, dim=1)


class WordBaker(nn.Module):
    def __init__(self, dictionarySize, sparkleFactor, brainSize, howManyLayers, brainCellType='GRU', forgetfulness=0.1):
        super(WordBaker, self).__init__()
        
        self.dictionarySize = dictionarySize
        self.embedding = nn.Embedding(dictionarySize, sparkleFactor)
        
        rnn_cell = getattr(nn, brainCellType)
        
        self.rnn = rnn_cell(sparkleFactor, brainSize, howManyLayers, 
                            dropout=forgetfulness, batch_first=True)
        
        self.finalGuessLayer = nn.Linear(brainSize, dictionarySize)
        self.forgetfulness = nn.Dropout(forgetfulness)

    def forward(self, oneWordIn, bakersThought):
        shinyWords = self.embedding(oneWordIn)
        shinyWords = self.forgetfulness(shinyWords)
        
        oneWordOut, newThought = self.rnn(shinyWords, bakersThought)
        
        bestGuess = self.finalGuessLayer(oneWordOut.squeeze(1))
        
        return F.log_softmax(bestGuess, dim=1), newThought

class TranslatorBot9000(nn.Module):
    def __init__(self, wordMuncher, wordBaker):
        super().__init__()
        self.wordMuncher = wordMuncher
        self.wordBaker = wordBaker

    def forward(self, sourceLanguage, targetLanguage, cheatSheetRatio=0.5):
        howManyAtOnce = targetLanguage.shape[0]
        howLongIsTheWord = targetLanguage.shape[1]
        howManyWordsToGuessFrom = self.wordBaker.dictionarySize
        
        allTheGuesses = torch.zeros(howManyAtOnce, howLongIsTheWord, howManyWordsToGuessFrom).to(sourceLanguage.device)
        
        allTheThoughts, lastThought = self.wordMuncher(sourceLanguage)
        
        oneWordIn = targetLanguage[:, 0].unsqueeze(1)
        
        for letterIndex in range(1, howLongIsTheWord):
            oneWordOut, lastThought = self.wordBaker(oneWordIn, lastThought)
            
            allTheGuesses[:, letterIndex] = oneWordOut
            
            if torch.rand(1) < cheatSheetRatio:
                oneWordIn = targetLanguage[:, letterIndex].unsqueeze(1)
            else:
                oneWordIn = oneWordOut.argmax(1).unsqueeze(1)
                
        return allTheGuesses
