# WordleBot

I've built an AI bot to play **Wordle** using Deep Reinforcement Learning! 

WordleBot Stats:  
1. **100%** Accuracy  
2. **3.53** Avg Guesses  
3. **~30M** Games Playes

[Try WordleBot](https://huggingface.co/spaces/RylieWeaver/WordleBot)  

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

Wordle is a game run by the New York Times where the goal is to guess an unknown 5-letter word (which we will call the target word) in as few attempts as possible (with a hard cap at 6). After each guess, the characters in the guessed word are highlighted according to their presence in the target word:
- **Green**: in the target word at that location
- **Yellow**: in the target word but not at that location, and the total number of greens/yellows does not exceed the count in the target word (characters on the left are given priority to be marked yellow).
- **Grey**: not in the target word or otherwise did not fit green/yellow requirements

<p align="left">
  <img src="images/game_peril.png" alt="Wordle game" width="300"/>
</p>

Besides small changes made since its release, Wordle has a total vocabulary of 12,972 words, of which 2,315 are possible target words. With those vocab sizes and 6 possible guesses, the maximum possible number of unique Wordle games is:
$$12972^6 \times 2315 \approx 10^{24}.$$
(quite large).

### Existing Approaches

There are quite a few different attempts of others for Wordle, including the use of heuristics, other machine learning models, and even brute force algorithms that fully solved the game. Yes, you read that right; Wordle has been solved for the goal of minimizing the average number of guesses with 100% accuracy. The exact stats are 100% accuracy with an average of 3.42 guesses. By definition, there is no chance that WordleBot can beat that optimal algorithm, so why still make it? Firstly, WordleBot is, as far as I know, the most performant deep learning model for playing Wordle, and the insights gained can advise other deep learning enthusiasts on RL for game-playing. Secondly, it was quite fun! As said before, there are quite a few other attempts out there, but here are some significant ones that I've found (note that MinMean Entropy and Maximum Information Gain are equivalent): 
- Greedy MinMean Entropy: [Link1](https://jluebeck.github.io/posts/WordleSolver) [Link2](https://nhsjs.com/wp-content/uploads/2024/04/Using-Information-Theory-to-Play-Wordle-as-Optimally-as-Possible.pdf)  
- [2-Step Maximum Information Gain](https://www.youtube.com/watch?v=v68zYyaEmEA) (inspired me to make this WordleBot!)  
- [Heuristic + Rollout](https://arxiv.org/pdf/2211.10298)  
- [Various Heuristic Comparisons](https://arxiv.org/pdf/2408.11730)  
- [Decision Trees](https://jonathanolson.net/experiments/optimal-wordle-solutions)  
- [Deep Leaning](https://andrewkho.github.io/wordle-solver/) (inspired WordleBot's Guess-State Attention!)  
- Optimal Strategy: [Link1](https://auction-upload-files.s3.amazonaws.com/Wordle_Paper_Final.pdf) [Link2](https://sonorouschocolate.com/notes/index.php/)  

### Reinforcement Learning

In Reinforcement Learning, a model takes in an input state, produces an action, and then receives a reward/penalty that gives feedback on that action. Despite its simple core RL has made a large impact on many machine learning application areas, including modern LLMs (large language models) lke ChatGPT. In particular, RL has been used to develop world-class or even superhuman level performance at varios games, including Chess, Go, DOTA, and Starcraft. Deep Reinforcement Learning uses deep neural networks as the core model architecture, and particularly shines over simpler methods (such as heuristics) when the action space and/or game length is exceedingly large. I was inspired by the success of these models and wanted to better understand how they work, so decided to make a bot on the much simpler game of Wordle.  


## WordleBot

### State and Action Representation

The state that is given WordleBot is the concatenation of the known information (alphabet state) and guess number (guess state). 
#### 1. Alphabet State (size 286 = 26 × 11)
- Each of the 26 letters in the English alphabet is represented with 11 features:
  - 5 for positions where the letter is known to occur
  - 5 for positions where the letter is known to not occur
  - 1 for the minimum number of known occurrences of the letter
#### 2. Guess State (size 6)
- A one-hot vector representing the current guess number (1 through 6), which tells WordleBot how far along in the game it is.

Note that many people would encode 15 features per letter (5 positions x 3 colors: green, yellow, grey). However, this state representation has significant crossover information that I would rather have compressed. A couple examples are:
- Grey implies "letter nowhere", regardless of the location it's observed.
- Both yellow and grey at a location imply "letter not here"
- Green implies both "letter here" and "all other letters not here"
This 11-feature representation compresses the state size while keeping all the same information.

---

#### 3. Action (size 130 = 26 × 5)
- Each action corresponds to a guessing a 5-letter word.
- The concatentation of five one-hot vectors, one for each position (26 possible letters × 5 positions).


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




[WordleBot GitHub Repo](https://github.com/RylieWeaver/WordleBot)  

My Contacts: LinkedIn(link)  |  Email: rylieweaver9@gmail.com  |  [GitHub Repo](https://github.com/RylieWeaver/WordleBot)  











