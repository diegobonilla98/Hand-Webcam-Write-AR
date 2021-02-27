# -*- coding: utf-8 -*-
"""
Created on Tue May  1 16:14:49 2018

@author: rcabg
"""
import sys
sys.path.append('./SpanishSpellingCorrector/')

class SpanishWordFreq:
    def __init__(self, filePath):
        self.file = open(filePath, "r", encoding = 'latin-1')
        self.words = {}
        self.totalFreq = 0
        self.parseFile()
        
    def parseFile(self):
        next(self.file)
        for line in self.file:
            index = line.find(".")
            try:
                editedLine = line[index+2:]
                editedLine = editedLine.replace(',', '')
                lineSplit = editedLine.split()
                self.words[lineSplit[0]] = int(lineSplit[1])
                self.totalFreq += int(lineSplit[1])
            except Exception:
                print("It seems there is a problem with the parsed line. Check it out:")
                print(editedLine)
                print(lineSplit)