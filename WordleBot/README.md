# Header

GitHub Link: https://github.com/RylieWeaver/WordleBot


## Summary

I've made a an AI bot to play the Wordle using Deep Reinforcement Learning! It was trained for ~30M games and gets the correct word 100% of the time with an average of 3.55 guesses (and still improving). \ 
Please do test it out here(https://huggingface.co/spaces/RylieWeaver/WordleBot) and share it with anyone who may be interested as well! At a high level, WordleBot is an A2C (Advantage Actor Critic) neural \
network trained with PPO (Proximal Policy Optimization). In addition, the following are the distinguishing features of WordleBot and how it was trained that are both impactful to performance and outside the \
basic norm for RL (Reinforcement Learning) systems:
(1) Inductive Bias: WordleBot is constrained to take/not take actions that are clearly optimal or suboptimal, respectively.
(2) Guess-State Attention: Compute the action probabilities as attention values between the state embedding and the action space embeddings.
(3) Expected Entropy Gain as Reward Function: Define rewards as the average entropy gain over 'm' possible target words ('m' is set as a hyperparameter).


## Background

### Wordle

Wordle is a game published by the New York Times, where the goal is to guess a certain target word given 6 guesses. After each guess, the user gets feedback ***. There are ~13k allowed guess words, and 2315 possible \
target words. This makes the total number of possible Wordle games = ***.

### Existing Approaches

There are quite a few different attempts of others for Wordle, including heuristic solvers and machine learning models. Here are some:
Heuristic:
- minmax or meanmax entropy greedy
- meanmax overall
- deep learning github post (this one inspired me in particular)

### Deep Reinforcement Learning
Deep Reinforcement Learning is the field that combines deep neural networks with reinforcement learning training. Some interesting areas this has made impact are ChatGPT, AlphaZero(Link), and ***. In particular, \
reinforcement learning has shown to make superhuman performance in a variety of games (Chess, Go, ***). At fundamental level, in RL, a model takes in an input state, produces an action, and then receives a reward/penalty \
that gives feedback on that action. I was inspired by the success of these models and wanted to better understand how they work, so decided to make a bot of my own on a much simpler game.


## WordleBot

### State and Action Representation

State:
The state for Wordle that is given WordleBot is size 292=(26*11)+6. This comes from the fact that we have 11 pieces of information for 26 letters in the alphabet, plus 6 total possible guesses in the game. The 11 pieces of \
information breaks down as follows: 5 of them indicate places we know the letter IS in the target word, 5 of them indicate places we know the letter ISN't in the taret word, and 1 indicates the minimum number of times we know \
that letter appears in the target word.

Action:
The possible actions for Wordle are all 5-letter words where each letter comes from a 26-letter alphabet. The action representation is resultingly a one-hot encoded as a one-hot encoding of the letter-action pairs with size \
130=(26*5), with 5 of indices in the one hot vector with a '1' and all others as '0'.

### Inductive Bias

There are certain optimal/suboptimal actions that are clear given a certain state. The arise in the form of rules that WordleBot's actions are constrained to follow. Specifically, the inductive biases that WordleBot follows are:
- If there is only one possible target word given the state, then WordleBot should guess that word.
- If WordleBot is on the last guess, then guess one of the possible target words.
- If WordleBot has already guessed a certain word, then don't guess it again.

By constraining WordleBot to follow these rules, we significantly decrease the size of the learning task that we give to WordleBot. In addition, we implement a KL-Div loss between the constrained and output probabilities of \
so that the parameters of WordleBot do receive a signal to follow these given rules.

### Guess-State Attention

We compute the probability vector for the action space as:

P = softmax_T(phi_1(S) phi_2(A) / sqrt(d))

Note that this is exactly the same formula for attention weights in Transformers if T=1.

### Reward Function

The reward function is set as the average entropy gain over 'm' possible target words where 'm' is set as a hyperparameter.








For example such as choosing a given word when it is the only possible target, or not 






My Contacts: LinkedIn(link)  |  Email: rylieweaver9@gmail.com
