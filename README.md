# Hand-Webcam-Write-AR
Write using just one finger!

I've used the mediapipe hand detection to train a Residual Neural Network to detect characters. Just lift your index finger to write in the air and see how it magically transforms into a character.
Hand detection is from [mediapipe](https://google.github.io/mediapipe/solutions/hands.html).
The project also implements a [grammar correction pipeline](https://github.com/rcabg/Spanish-Spelling-Corrector) to autocorrect misspelled (although it's only available in Spanish, for english check out [this library](https://pypi.org/project/pyspellchecker/)). 

## Demo
In my linkedin [profile](https://www.linkedin.com/feed/update/urn:li:activity:6772408346923294720/?commentUrn=urn%3Ali%3Acomment%3A(ugcPost%3A6772408254229184512%2C6773764301196021760))

## Characters
The following image contains all possible characters the network is learnt to distinguish and some ways you can write them. Also, try to replicate the shape as much as possible for better results.

![](alphabet.png)

## Model
The repostery includes all necessary files to train the NN. Probably overfits since there is not enough data, but life is pretty complicated. The included trained model has an accuracy +-78% which for me works fine.

## Requirements
- Python 3.7
- Tensorflow 1.15
- OpenCV
- Mediapipe
- Pyautogui
- idk, just look at the imported libraries...

## Disclaimer
This was a cool idea made entirely in 1 day for learning purposes only. This is not a product of any kind and doen't have to work 100% of the time. Anyone can get the code and modify it **and only if they want, mention my name** but not required to do it. Cheers. 
