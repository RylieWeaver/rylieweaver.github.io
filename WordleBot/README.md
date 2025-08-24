# WordleBot

I've built an AI bot to play **Wordle** using Deep Reinforcement Learning!  

[Try Out WordleBot](https://huggingface.co/spaces/RylieWeaver/WordleBot) (please share it with anyone who may be interested!)  
[GitHub Repo](https://github.com/RylieWeaver/WordleBot)  

WordleBot was trained for **~30M** games and gets the correct word **100% of the time** with an average of **3.55 guesses** (and still improving).  

---

## Summary

At a high level, WordleBot is an A2C (Advantage Actor-Critic) neural network trained with PPO (Proximal Policy Optimization). In addition, the following features distinguish WordleBot and its training from more standard RL (Reinforcement Learning) systems:

1. Inductive Bias:  
   WordleBot is constrained to take/not take actions that are clearly optimal/suboptimal, respectively.

2. Guess-State Attention:  
   WordleBot computes its action probabilities as attention values between the state embedding and the action embeddings.  

3. Expected Entropy Gain as Reward Function:  
   Rewards are defined as the expected entropy gain over the target words.  


## Background

### Wordle

Wordle is a game published by the New York Times, where the goal is to guess a certain target word given 6 guesses. After each guess, the user gets feedback ***. There are ~13k allowed guess words, and 2315 possible \
target words. This makes the total number of possible Wordle games = ***.

Wordle is a game run by the New York Times where the goal is to guess an unknown 5-letter word (which we will call the target word) in as few attempts as possible (with a hard cap at 6). After each guess, the player receives feedback from the game, which comes in the form of color highlighting for each of the characters in the guessed word. For each character, it (i) turns grey if it is not in the target word, (ii) turns green if it is in the target word at that location, and (iii) turns yellow if it is in the target word, but not at that location (aside from the following exception). The one exception where a letter may not turn yellow despite fitting the description of (iii) is when a letter appears more times in the guessed word than the target word. In such a case, the count of greens/yellows is capped by the number of occurences of that letter in the target word, with greens/leftmost characters being given priority. For example, if the target word is "model" and the guess word was "mummy", only the leftmost "m" would turn green and all other occurrences gray. Similarly, if the target word was "model" and the guess word was "dummy", only the leftmost "m" would appear yellow and all others gray. 

An example Wordle game is pictured in  Figure~\ref{fig:peril} below, where the target word "peril" is successfully guessed after three moves.

\begin{figure}[h]
    \centering
    \includegraphics[width=0.3\linewidth]{images/peril.png}
    \caption{Illustration of a Wordle game with the target word peril.}
    \label{fig:peril}
\end{figure}




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

We compute WordleBot's probability vector for the action space as:

P = softmax_T(phi_1(S) phi_2(A) / sqrt(d))

Note that this is exactly the same formula for attention weights in Transformers if T=1.

### Reward Function

The reward function is set as the average entropy gain over 'm' possible target words where 'm' is set as a hyperparameter. So, given a state S, the total possible target vocab T, and a set of M-subset-V possible target words \
given the state S, then:

R = 1/m sum[ entropy(S) - entropy(S+1 | t=m_i)]

where m_i are sample from M (without replacement if m >= |M| and with replacement is m <= |M|).








For example such as choosing a given word when it is the only possible target, or not 






My Contacts: LinkedIn(link)  |  Email: rylieweaver9@gmail.com


